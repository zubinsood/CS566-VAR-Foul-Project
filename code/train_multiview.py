# Multi-view training script for MV-Foul

import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from mvfoul_multiview_dataset import MVFoulMultiViewDataset
from mvfoul_multiview_model import MVFoulMultiViewBaseline


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[train-mv] Using MPS device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[train-mv] Using CUDA device")
        return torch.device("cuda")
    else:
        print("[train-mv] Using CPU")
        return torch.device("cpu")


def compute_balanced_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Balanced accuracy = mean of per-class recall
    """
    eps = 1e-8
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for t, p in zip(y_true.cpu(), y_pred.cpu()):
        conf[t, p] += 1

    per_class_recall = []
    for c in range(num_classes):
        tp = conf[c, c].float()
        total = conf[c].sum().float()
        recall = tp / (total + eps)
        per_class_recall.append(recall.item())

    return float(sum(per_class_recall) / len(per_class_recall))


def build_datasets(
    data_root: str,
    num_frames: int,
    num_views: int,
    img_size: int = 224,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    """
    Build train and valid datasets + transforms for multi-view
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])

    train_dataset = MVFoulMultiViewDataset(
        root=data_root,
        split="train",
        num_frames=num_frames,
        num_views=num_views,
        transform=transform,
    )

    val_dataset = MVFoulMultiViewDataset(
        root=data_root,
        split="valid",
        num_frames=num_frames,
        num_views=num_views,
        transform=transform,
    )

    if max_train_samples is not None:
        max_train = min(max_train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(max_train)))
        print(f"[train-mv] Using {max_train} train samples (subset)")

    if max_val_samples is not None:
        max_val = min(max_val_samples, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(max_val)))
        print(f"[train-mv] Using {max_val} val samples (subset)")

    return train_dataset, val_dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_sev: nn.Module,
    criterion_type: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Train for one epoch (multi-view).
      clips: (B, V, T, C, H, W)
    """
    model.train()
    total_loss = 0.0
    total_sev_loss = 0.0
    total_type_loss = 0.0
    n_samples = 0

    for clips, sev_labels, type_labels in loader:
        clips = clips.to(device)             # (B, V, T, C, H, W)
        sev_labels = sev_labels.to(device)   # (B,)
        type_labels = type_labels.to(device) # (B,)

        optimizer.zero_grad()

        sev_logits, type_logits = model(clips)

        loss_sev = criterion_sev(sev_logits, sev_labels)
        loss_type = criterion_type(type_logits, type_labels)
        loss = loss_sev + loss_type

        loss.backward()
        optimizer.step()

        bs = clips.size(0)
        total_loss += loss.item() * bs
        total_sev_loss += loss_sev.item() * bs
        total_type_loss += loss_type.item() * bs
        n_samples += bs

    avg_loss = total_loss / n_samples
    avg_sev_loss = total_sev_loss / n_samples
    avg_type_loss = total_type_loss / n_samples

    return avg_loss, avg_sev_loss, avg_type_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_sev: nn.Module,
    criterion_type: nn.Module,
    device: torch.device,
    num_foul_types: int,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate on validation set
    Returns:
      avg_loss, avg_sev_loss, avg_type_loss,
      bal_acc_sev, bal_acc_type
    """
    model.eval()
    total_loss = 0.0
    total_sev_loss = 0.0
    total_type_loss = 0.0
    n_samples = 0

    all_sev_true = []
    all_sev_pred = []
    all_type_true = []
    all_type_pred = []

    with torch.no_grad():
        for clips, sev_labels, type_labels in loader:
            clips = clips.to(device)
            sev_labels = sev_labels.to(device)
            type_labels = type_labels.to(device)

            sev_logits, type_logits = model(clips)

            loss_sev = criterion_sev(sev_logits, sev_labels)
            loss_type = criterion_type(type_logits, type_labels)
            loss = loss_sev + loss_type

            bs = clips.size(0)
            total_loss += loss.item() * bs
            total_sev_loss += loss_sev.item() * bs
            total_type_loss += loss_type.item() * bs
            n_samples += bs

            sev_pred = sev_logits.argmax(dim=1)
            type_pred = type_logits.argmax(dim=1)

            all_sev_true.append(sev_labels.cpu())
            all_sev_pred.append(sev_pred.cpu())
            all_type_true.append(type_labels.cpu())
            all_type_pred.append(type_pred.cpu())

    avg_loss = total_loss / n_samples
    avg_sev_loss = total_sev_loss / n_samples
    avg_type_loss = total_type_loss / n_samples

    all_sev_true = torch.cat(all_sev_true, dim=0)
    all_sev_pred = torch.cat(all_sev_pred, dim=0)
    all_type_true = torch.cat(all_type_true, dim=0)
    all_type_pred = torch.cat(all_type_pred, dim=0)

    bal_acc_sev = compute_balanced_accuracy(all_sev_true, all_sev_pred, num_classes=4)
    bal_acc_type = compute_balanced_accuracy(
        all_type_true, all_type_pred, num_classes=num_foul_types
    )

    return avg_loss, avg_sev_loss, avg_type_loss, bal_acc_sev, bal_acc_type


def main():
    parser = argparse.ArgumentParser(description="Train multi-view MV-Foul baseline")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to mvfouls root (the folder containing train/ and valid/)",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=256)
    parser.add_argument("--max_val_samples", type=int, default=256)
    parser.add_argument("--use_pretrained", action="store_true")

    args = parser.parse_args()

    device = get_device()

    # 1) Datasets
    train_dataset, val_dataset = build_datasets(
        data_root=args.data_root,
        num_frames=args.num_frames,
        num_views=args.num_views,
        img_size=args.img_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    base_train_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    num_foul_types = len(base_train_dataset.foul_type_mapping)
    print(f"[train-mv] num_foul_types = {num_foul_types}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # 2) Model, loss, optimizer
    model = MVFoulMultiViewBaseline(
        num_foul_types=num_foul_types,
        use_pretrained=args.use_pretrained,
    ).to(device)

    criterion_sev = nn.CrossEntropyLoss()
    criterion_type = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("results", exist_ok=True)
    best_val_loss = float("inf")
    best_model_path = os.path.join("results", "multiview_best.pth")

    # 3) Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n----- Epoch {epoch}/{args.epochs} -----")

        train_loss, train_sev_loss, train_type_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_sev,
            criterion_type,
            device,
        )

        print(
            f"[train-mv] loss={train_loss:.4f}, "
            f"sev_loss={train_sev_loss:.4f}, "
            f"type_loss={train_type_loss:.4f}"
        )

        (
            val_loss,
            val_sev_loss,
            val_type_loss,
            bal_acc_sev,
            bal_acc_type,
        ) = evaluate(
            model,
            val_loader,
            criterion_sev,
            criterion_type,
            device,
            num_foul_types=num_foul_types,
        )

        print(
            f"[valid-mv] loss={val_loss:.4f}, "
            f"sev_loss={val_sev_loss:.4f}, "
            f"type_loss={val_type_loss:.4f}, "
            f"bal_acc_sev={bal_acc_sev:.4f}, "
            f"bal_acc_type={bal_acc_type:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_foul_types": num_foul_types,
                },
                best_model_path,
            )
            print(f"[train-mv] New best model saved to {best_model_path}")


if __name__ == "__main__":
    main()
