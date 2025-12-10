import torch
import torch.nn as nn
from torchvision.models import resnet18


class MVFoulMultiViewBaseline(nn.Module):
    """
    Multi-view baseline model for MV-Foul

    Input:
        x: (B, V, T, C, H, W)
            B = batch size
            V = number of views
            T = frames per view
            C, H, W = channels, height, width

    Architecture:
        - Apply ResNet18 frame-wise to all frames of all views
        - Average features over time per view -> (B, V, F)
        - Average over views -> (B, F)
        - Two heads:
            * severity_head: 4 classes (0..3)
            * foul_type_head: num_foul_types classes
    """

    def __init__(
        self,
        num_foul_types: int,
        use_pretrained: bool = False,
    ):
        super().__init__()

        # 2D CNN backbone applied frame-wise
        self.backbone = resnet18(weights="IMAGENET1K_V1" if use_pretrained else None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove final classification layer

        # Heads
        self.severity_head = nn.Linear(feat_dim, 4)  # 4 severity classes (0..3)
        self.foul_type_head = nn.Linear(feat_dim, num_foul_types)

    def forward(self, x: torch.Tensor):
        """
        x: (B, V, T, C, H, W)
        returns:
          severity_logits: (B, 4)
          foul_type_logits: (B, num_foul_types)
        """
        B, V, T, C, H, W = x.shape

        # (B, V, T, C, H, W) -> (B*V*T, C, H, W)
        x = x.view(B * V * T, C, H, W)

        # Frame-wise features
        feats = self.backbone(x)  # (B*V*T, F)

        # (B*V*T, F) -> (B, V, T, F)
        Fdim = feats.shape[-1]
        feats = feats.view(B, V, T, Fdim)

        # Average over time per view: (B, V, F)
        feats_per_view = feats.mean(dim=2)

        # Average over views: (B, F)
        clip_feat = feats_per_view.mean(dim=1)

        # Heads
        severity_logits = self.severity_head(clip_feat)
        foul_type_logits = self.foul_type_head(clip_feat)

        return severity_logits, foul_type_logits