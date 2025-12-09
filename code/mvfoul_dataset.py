import os
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import av  # Video Reading package
except ImportError:
    av = None
    print("[MVFoulDataset] ERROR: pyav is not installed, install with `pip install av`")


class MVFoulDataset(Dataset):
    """
    PyTorch Dataset for SoccerNet MV-Foul (single-view baseline)

    Expected directory structure:

        root/
          train/
            annotations.json
            action_0/
              *.mp4
            action_1/
              *.mp4
            ...
            action_2915/
              *.mp4

          valid/
            annotations.json
            action_0/
              *.mp4
            action_1/
              *.mp4
            ...
            action_410/
              *.mp4

    Notes:
      - Each split (train/valid) has its own zero-based action indexing:
        - train: 2916 actions (0–2915)
        - valid: 411 actions (0–410)
      - This version loads a single view per action (the first .mp4 in the folder)
        and returns a clip tensor of shape (T, C, H, W) plus severity and foul type labels
      - It is used as the single-view baseline dataset; will extend it later
        to support multi-view clips for each action
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 32,
        transform=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert split in ["train", "valid"], f"Unknown split: {split}"

        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.device = device

        self.split_dir = os.path.join(root, split)
        self.ann_path = os.path.join(self.split_dir, "annotations.json")

        # Initial mapping for some common action classes (type-of-foul)
        # This dictionary is **expanded dynamically** in _parse_foul_type() whenever a new 'Action class' label is encountered in the annotations
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
        print(f"[MVFoulDataset] Loaded {len(self.samples)} {split} samples")

    # Annotation Loading
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load annotations.json and build a list of samples

        "annotations.json" structure (from inspect_annotations):

            {
              "Set": "Train",
              "Number of actions": 2916,
              "Actions": {
                "0": {
                  "UrlLocal": "...",
                  "Offence": "Offence",
                  "Contact": "...",
                  "Action class": "Challenge",
                  "Severity": "1.0",
                  "Clips": [
                    {"Url": "Dataset/Train/action_0/clip_0", ...},
                    {"Url": "Dataset/Train/action_0/clip_1", ...}
                  ]
                },
                "1": { ... },
                ...
              }
            }

        Notes:
          - iterate over data["Actions"].items()
          - map key "0" -> action directory "action_0"
          - parse severity and foul-type labels
        """
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
            # key: "0", "1", etc.
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

    # Label parsing helpers
    def _parse_severity(self, ann: Dict[str, Any]) -> int:
        """
        Parse offence severity label and map to [0..3]

        Heuristic:
          - If Offence == "No offence" -> 0
          - Else use the numeric Severity field:
                1.0 -> 1
                2.0 -> 2
                3.0 -> 3
        """
        offence = ann.get("Offence", "")
        severity_str = ann.get("Severity", "0.0")

        if offence == "No offence":
            return 0

        try:
            sev = int(float(severity_str))
        except Exception:
            # If parsing fails, treat as "no offence"
            sev = 0

        # Clamp to [0, 3]
        sev = max(0, min(3, sev))
        return sev

    def _parse_foul_type(self, ann: Dict[str, Any]) -> int:
        """
        Parse 'Action class' -> integer label

        If there is an unseen action class, add it dynamically to
        "self.foul_type_mapping"
        """
        cls = ann.get("Action class", None)
        if cls is None:
            cls = "Unknown"

        if cls not in self.foul_type_mapping:
            # Dynamically add unknown types
            new_id = len(self.foul_type_mapping)
            print(f"[MVFoulDataset] New action class '{cls}' -> id {new_id}")
            self.foul_type_mapping[cls] = new_id

        return self.foul_type_mapping[cls]

    # Video loading helpers
    def _get_first_view_path(self, action_dir: str) -> str:
        """
        Return the path to the first .mp4 file in the action directory

        Can extend this to:
          - sort views by name
          - load multiple views
        """
        if not os.path.isdir(action_dir):
            raise FileNotFoundError(f"Action directory not found: {action_dir}")

        vids = [f for f in os.listdir(action_dir) if f.lower().endswith(".mp4")]
        vids.sort()
        if len(vids) == 0:
            raise FileNotFoundError(f"No .mp4 files found in {action_dir}")

        return os.path.join(action_dir, vids[0])

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Load frames from the given video using pyav, sample self.num_frames
        uniformly, and return a tensor of shape (T, C, H, W) in float32 [0,1]
        """
        if av is None:
            raise ImportError(
                "pyav is not installed, run 'pip install av' in environment"
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
        # Use linspace to sample indices (with repetition if too short)
        indices = torch.linspace(0, num_total - 1, steps=self.num_frames).long()
        frames = frames[indices]

        if self.transform is not None:
            # Apply transform per frame
            frames = torch.stack([self.transform(f) for f in frames], dim=0)

        if self.device is not None:
            frames = frames.to(self.device)

        return frames  # (T, C, H, W)

    # Dataset Interface
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]

        video_path = self._get_first_view_path(sample["action_dir"])
        clip = self._load_video_frames(video_path)

        severity = sample["severity"]
        foul_type = sample["foul_type"]

        return clip, severity, foul_type