# code/mvfoul_multiview_dataset.py

import os
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import av
except ImportError:
    av = None
    print("[MVFoulMultiViewDataset] ERROR: pyav is not installed, install with 'pip install av'")


class MVFoulMultiViewDataset(Dataset):
    """
    PyTorch Dataset for SoccerNet MV-Foul (multi-view version)

    For each action, this dataset:
      - finds all .mp4 clips in action_<id>/
      - loads up to "num_views" views
      - samples "num_frames" per view
      - returns a tensor of shape (V, T, C, H, W)
        where V = number of views actually loaded (<= num_views)

    Labels:
      - severity: 0..3
      - foul_type: integer label (same mapping as single-view version)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        num_views: int = 3,
        transform=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert split in ["train", "valid"], f"Unknown split: {split}"

        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.num_views = num_views
        self.transform = transform
        self.device = device

        self.split_dir = os.path.join(root, split)
        self.ann_path = os.path.join(self.split_dir, "annotations.json")

        # Same initial mapping as single-view dataset
        self.foul_type_mapping = {
            "Standing tackling": 0,
            "Tackling": 1,
            "High leg": 2,
            "Pushing": 3,
            "Holding": 4,
            "Elbowing": 5,
            "Challenge": 6,
            "Dive/Simulation": 7,
        }

        self.samples = self._load_annotations()
        print(f"[MVFoulMultiViewDataset] Loaded {len(self.samples)} {split} samples")

    # Annotation loading (same logic as single-view)
    def _load_annotations(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.ann_path):
            raise FileNotFoundError(f"annotations.json not found at {self.ann_path}")

        with open(self.ann_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "Actions" not in data:
            raise ValueError("Unexpected annotations.json format (no 'Actions' key)")

        actions = data["Actions"]
        if not isinstance(actions, dict):
            raise ValueError("Expected 'Actions' to be a dict of {id: annotation}")

        samples: List[Dict[str, Any]] = []

        for key, ann in actions.items():
            action_folder = f"action_{key}"
            action_dir = os.path.join(self.split_dir, action_folder)
            if not os.path.isdir(action_dir):
                continue

            severity_label = self._parse_severity(ann)
            foul_type_label = self._parse_foul_type(ann)

            samples.append(
                {
                    "key": key,
                    "action_dir": action_dir,
                    "severity": severity_label,
                    "foul_type": foul_type_label,
                }
            )

        return samples

    def _parse_severity(self, ann: Dict[str, Any]) -> int:
        offence = ann.get("Offence", "")
        severity_str = ann.get("Severity", "0.0")

        if offence == "No offence":
            return 0

        try:
            sev = int(float(severity_str))
        except Exception:
            sev = 0

        sev = max(0, min(3, sev))
        return sev

    def _parse_foul_type(self, ann: Dict[str, Any]) -> int:
        cls = ann.get("Action class", None)
        if cls is None:
            cls = "Unknown"

        if cls not in self.foul_type_mapping:
            new_id = len(self.foul_type_mapping)
            print(f"[MVFoulMultiViewDataset] New action class '{cls}' -> id {new_id}")
            self.foul_type_mapping[cls] = new_id

        return self.foul_type_mapping[cls]

    # Video loading helpers
    def _get_view_paths(self, action_dir: str) -> List[str]:
        """
        Return a sorted list of all .mp4 files in this action directory
        We will then take up to "self.num_views" of them
        """
        if not os.path.isdir(action_dir):
            raise FileNotFoundError(f"Action directory not found: {action_dir}")

        vids = [f for f in os.listdir(action_dir) if f.lower().endswith(".mp4")]
        vids.sort()
        if len(vids) == 0:
            raise FileNotFoundError(f"No .mp4 files found in {action_dir}")

        full_paths = [os.path.join(action_dir, v) for v in vids]
        return full_paths

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Load frames from the given video using pyav, sample "self.num_frames"
        uniformly, and return a tensor of shape (T, C, H, W) in float32 [0,1]
        """
        if av is None:
            raise ImportError(
                "pyav is not installed, run 'pip install av' in this environment"
            )

        container = av.open(video_path)
        frames = []

        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()  # (H, W, 3), uint8
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
            frames.append(tensor)

        container.close()

        if len(frames) == 0:
            raise ValueError(f"No frames decoded from {video_path}")

        frames = torch.stack(frames, dim=0)  # (num_total_frames, 3, H, W)
        num_total = frames.shape[0]

        indices = torch.linspace(0, num_total - 1, steps=self.num_frames).long()
        frames = frames[indices]

        if self.transform is not None:
            frames = torch.stack([self.transform(f) for f in frames], dim=0)

        if self.device is not None:
            frames = frames.to(self.device)

        return frames  # (T, C, H, W)

    # Dataset interface
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        action_dir = sample["action_dir"]

        view_paths = self._get_view_paths(action_dir)
        # Limit to num_views
        view_paths = view_paths[: self.num_views]

        clips = []
        for vp in view_paths:
            clip = self._load_video_frames(vp)  # (T, C, H, W)
            clips.append(clip)

        # (V, T, C, H, W)
        clips = torch.stack(clips, dim=0)

        severity = sample["severity"]
        foul_type = sample["foul_type"]

        return clips, severity, foul_type