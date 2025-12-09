# test_model_shapes.py
import torch
from mvfoul_dataset import MVFoulDataset
from mvfoul_model import MVFoulBaseline


def main():
    """
    Quick shape test:
      - Load a tiny MVFoulDataset
      - Extract num_foul_types from its mapping
      - Build the MVFoulBaseline
      - Run a forward pass on a dummy clip with correct shapes
    """

    # Load dataset (train split) to infer label dimensions
    # root = "/path/to/SoccerNetData/mvfouls"
    root = "/Users/zubin/Desktop/CS566/SoccerNetData/mvfouls"
    dataset = MVFoulDataset(root=root, split="train", num_frames=16)

    num_foul_types = len(dataset.foul_type_mapping)

    # Print for confirmation
    print(f"[test] Dataset foul types: {num_foul_types}")

    # Create model
    model = MVFoulBaseline(num_foul_types=num_foul_types, use_pretrained=False)

    # Build dummy input with the same spatial size as real data
    # Use the first sample to get correct H, W
    sample_clip, _, _ = dataset[0]     # (T, C, H, W)
    T, C, H, W = sample_clip.shape

    B = 2
    dummy_clip = torch.randn(B, T, C, H, W)

    # Forward pass
    sev_logits, type_logits = model(dummy_clip)

    print("Severity logits shape:", sev_logits.shape)       # expect (B, 4)
    print("Foul type logits shape:", type_logits.shape)     # expect (B, num_foul_types))


if __name__ == "__main__":
    main()