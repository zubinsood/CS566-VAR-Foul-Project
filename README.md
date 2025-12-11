# 📘 CS566 — VAR Foul Recognition (SoccerNet MV-Foul)
[Presentation](https://youtu.be/SjkTu-eWUXA)

This repository contains our final project for **CS566: Computer Vision**, where we implement and experiment with the **SoccerNet Multi-View Foul Recognition** task.

Our work includes:

- dataset preprocessing  
- a complete **single-view baseline** (ResNet18)  
- a full **multi-view fusion baseline** (2-view average fusion)  
- training & evaluation pipeline  
- utilities for inspecting annotations  
- result logs  
- a final project website  
---

## 📁 Repository Structure

    CS566-VAR-Foul-Project/
    │
    ├── code/
    │   ├── mvfoul_dataset.py               # Single-view dataset
    │   ├── mvfoul_model.py                 # Single-view model
    │   ├── train_baseline.py               # Train single-view model
    │   ├── mvfoul_multiview_dataset.py     # Multi-view dataset
    │   ├── mvfoul_multiview_model.py       # Multi-view fusion model
    │   ├── train_multiview.py              # Train multi-view model
    │   ├── test_dataset.py                 # Single-view dataset check
    │   ├── test_model_shapes.py            # Single-view model check
    │   ├── test_multiview_dataset.py       # Multi-view dataset check
    │   ├── test_multiview_model_shapes.py  # Multi-view model check
    │   ├── inspect_annotations.py          # Annotation inspector
    │   └── ...
    │
    ├── scripts/
    │   ├── download_train.py
    │   ├── download_valid.py
    │   └── ...
    │
    ├── results/
    │   ├── baseline_run1.md
    │   ├── multiview_run1.md
    │   └── (model checkpoints ignored by git)
    │
    ├── figures/          # Plots / visualizations
    │
    ├── website/          # Final project webpage
    │
    ├── README.md
    └── .gitignore

---

## 👥 Team Members

- **Zubin Sood**  
- **Rithvik Banda**  
- **Jamil Kazimzade**

---

## 🎯 Project Overview

The task is to classify soccer video clips into:

### 1. Foul Severity (4 classes)

- 0 — No Offence  
- 1 — Offence + No Card  
- 2 — Offence + Yellow  
- 3 — Offence + Red  

### 2. Foul Type (8–11 classes)

From annotations, including: Tackling, Standing Tackling, High Leg, Holding, Pushing, Elbowing, Challenge, Dive, etc.

Class count varies slightly due to real annotations, so our loader **dynamically expands** classes when new “Action class” labels appear.

**Main metric:** Balanced Accuracy (mean recall across classes).

---

## 📦 Downloading the SoccerNet MV-Foul Dataset

1️⃣ **Sign the SoccerNet NDA**  
Required to access MV-Foul videos:  
https://www.soccer-net.org/data  

2️⃣ **Install the SoccerNet API**

    pip install SoccerNet --upgrade

3️⃣ **Download the train/valid splits**

    python3 scripts/download_train.py
    python3 scripts/download_valid.py

Before running these scripts, edit them and set:

    data_dir = "/path/to/SoccerNetData"

---

# ✅ Single-View Baseline (ResNet18)

**Setup:**

- Train samples: 1000  
- Validation samples: 300  
- Frames per clip: 16  
- Batch size: 4  
- Epochs: 5  
- Backbone: ResNet18 (ImageNet pretrained)  
- Device: MPS (Mac)  

### Best Validation Window (Epochs 1–3)

| Epoch | Val Loss | Severity BalAcc | Foul Type BalAcc |
|-------|----------|------------------|-------------------|
| 1     | 3.0287   | 0.2430           | 0.0855            |
| 2     | 3.1253   | 0.2913           | 0.0884            |
| 3     | 3.2278   | 0.2864           | 0.0900            |

The model starts to overfit after epoch 3, so we treat epochs 1–3 as the **clean single-view baseline**.

---

## ▶️ Run Single-View Training

    python3 code/train_baseline.py \
      --data_root "/path/to/SoccerNetData/mvfouls" \
      --epochs 5 \
      --batch_size 4 \
      --num_frames 16 \
      --max_train_samples 1000 \
      --max_val_samples 300 \
      --use_pretrained

This writes a checkpoint:

    results/baseline_best.pth   (ignored by git)

A summary of this run is logged in:

- `results/baseline_run1.md`

---

# 🎥 Multi-View Baseline (2-View Average Fusion)

The multi-view baseline extends the single-view architecture to use **multiple camera views** per foul event:

1. Load multiple synchronized clips for each action (e.g., main camera + close-up).  
2. Pass each view through a shared ResNet18 backbone.  
3. Average the per-view features to obtain a fused clip descriptor.  
4. Feed the fused representation into two heads:
   - severity head (4 classes)  
   - foul type head (N classes from annotations)  

**Training setup:**

- Views per action: 2  
- Frames per view: 16  
- Train samples: 1000  
- Validation samples: 300  
- Batch size: 2  
- Epochs: 3  
- Backbone: ResNet18 (ImageNet pretrained)  

### Validation Performance (3 Epochs)

| Epoch | Val Loss | Severity BalAcc | Foul Type BalAcc |
|-------|----------|------------------|-------------------|
| 1     | 3.2196   | 0.2383           | 0.0810            |
| 2     | 3.1427   | 0.2737           | 0.0928            |
| 3     | 2.9233   | 0.2548           | 0.1091            |

The multi-view baseline shows **improved foul-type balanced accuracy** compared to the single-view model, supporting the intuition that using multiple camera views helps classify the type of foul.

---

## ▶️ Run Multi-View Training

    python3 code/train_multiview.py \
      --data_root "/path/to/SoccerNetData/mvfouls" \
      --epochs 3 \
      --batch_size 2 \
      --num_frames 16 \
      --num_views 2 \
      --max_train_samples 1000 \
      --max_val_samples 300 \
      --use_pretrained

This produces:

- `results/multiview_best.pth`   (ignored by git)  
- `results/multiview_run1.md`    (validation metrics per epoch)

---

## 🧪 Debug / Utility Scripts

**Inspect annotation file structure**

    python3 code/inspect_annotations.py

**Test single-view dataset loading**

    python3 code/test_dataset.py

**Test single-view model shapes**

    python3 code/test_model_shapes.py

**Test multi-view dataset loading**

    python3 code/test_multiview_dataset.py

**Test multi-view model shapes**

    python3 code/test_multiview_model_shapes.py

---

## 🌐 Final Project Website

TODO

---

v
## 🎨 Visual Interface

**Launch interactive web interface for model visualization**

    python3 code/gradio_interface.py \
      --data_root "data/mvfouls" \
      --checkpoint "results/baseline_best.pth" \
      --split valid \
      --port 7860

The interface allows you to:
- Browse through dataset samples with a slider
- See real-time predictions for foul severity and type
- Compare predictions with ground truth labels
- View video clips with overlaid predictions
- See probability distributions for all classes

**Requirements:**
- Install Gradio: `pip install gradio`
- Optional: Install OpenCV for video overlay: `pip install opencv-python`

**Visual inference script (static images)**

    python3 code/visual_inference.py \
      --data_root "data/mvfouls" \
      --checkpoint "results/baseline_best.pth" \
      --split valid \
      --num_samples 5 \
      --save_dir "figures/inference"

---


## ✔ README Status

This README reflects the current state of the project:

- single-view baseline implemented and trained  
- multi-view baseline implemented and trained  
- result logs for both baselines  
- clear instructions to download data and rerun experiments  
- code / scripts layout
