import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MVFoulBaseline(nn.Module):
    """
    Simple baseline model for MV-Foul

    Input:  clip of shape (B, T, C, H, W)
    Steps:
      - reshape to (B*T, C, H, W)
      - pass through ResNet18 backbone (no pretrain or with pretrain if desired)
      - average features over time -> (B, F)
      - two heads:
          severity_head: 4 classes
          foul_type_head: num_foul_types classes
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
        self.backbone.fc = nn.Identity() # remove final classification layer

        # Heads
        self.severity_head = nn.Linear(feat_dim, 4) # 4 severity classes (0..3)
        self.foul_type_head = nn.Linear(feat_dim, num_foul_types)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C, H, W)
        returns:
          severity_logits: (B, 4)
          foul_type_logits: (B, num_foul_types)
        """
        B, T, C, H, W = x.shape

        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Frame-wise features
        feats = self.backbone(x) # (B*T, F)

        # (B*T, F) -> (B, T, F)
        Fdim = feats.shape[-1]
        feats = feats.view(B, T, Fdim)

        # Average over time: (B, F)
        clip_feat = feats.mean(dim=1)

        # Heads
        severity_logits = self.severity_head(clip_feat)
        foul_type_logits = self.foul_type_head(clip_feat)

        return severity_logits, foul_type_logits