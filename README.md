# 📘 CS566 — VAR Foul Recognition (SoccerNet MV-Foul)

This repository contains our final project for **CS566: Computer Vision**, where we implement and experiment with the **SoccerNet Multi-View Foul Recognition** task.

Our work includes:

- dataset preprocessing  
- a complete **single-view baseline** (ResNet18)  
- training & evaluation pipeline  
- utilities for inspecting annotations  
- model checkpoints  
- project website summarizing our results  

Later in the project, we extend this to a **multi-view fusion model**.

---

## 📁 Repository Structure

    CS566-VAR-Foul-Project/
    │
    ├── code/                       # Dataset, model, training, debugging scripts
    │   ├── mvfoul_dataset.py       # Single-view dataset loader
    │   ├── mvfoul_model.py         # Single-view baseline model
    │   ├── train_baseline.py       # Training script
    │   ├── test_dataset.py         # Sanity check for dataset loading
    │   ├── test_model_shapes.py    # Sanity check for model forward pass
    │   ├── inspect_annotations.py  # Annotation structure inspector
    │   └── ...
    │
    ├── scripts/                    # Utility scripts (no training)
    │   ├── download_train.py
    │   ├── download_valid.py
    │   └── ...
    │
    ├── results/                    # Saved checkpoints, logs, future plots
    │   └── baseline_best.pth
    │
    ├── figures/                    # Plots + images for final webpage
    │
    ├── website/                    # Final project webpage (index.html + assets)
    │
    ├── README.md
    └── .gitignore

---

## 👥 Team Members

**Zubin Sood**  
**Rithvik Banda**  
**Jamil Kazimzade**

---

## 🎯 Project Overview

The objective is to predict:

- **Foul Severity** (0–3)  
- **Foul Type** (11 classes, dynamically discovered in annotations)

using multi-view video clips from the **SoccerNet MV-Foul** dataset.

Our project involves:

1. Recreating the **baseline single-view model** described in MVNetwork.  
2. Extending it to a **multi-view fusion architecture**.  
3. Running controlled experiments comparing:
   - pretrained vs non-pretrained backbones  
   - different fusion approaches  
   - hyperparameter variations  
4. Publishing a polished project webpage summarizing:
   - dataset insights  
   - model architecture diagrams  
   - performance metrics  
   - visualizations  
   - downloadable code & checkpoints  

---

## 📦 Downloading the SoccerNet MV-Foul Dataset

### 1️⃣ Sign the SoccerNet NDA  

Required for access to MV-Foul videos.  
Apply via: https://www.soccer-net.org/data  

### 2️⃣ Install the SoccerNet API  

    pip install SoccerNet --upgrade

### 3️⃣ Download the dataset splits

    python3 scripts/download_train.py
    python3 scripts/download_valid.py

### 4️⃣ Set the dataset directory  

Before running the download scripts, edit them and set:

    data_dir = "/path/to/SoccerNetData"

---

# ✅ Baseline (Single-View ResNet18, Pretrained)

**Setup:**

- Train samples: **1000**  
- Val samples: **300**  
- Frames per clip: **16**  
- Batch size: **4**  
- Epochs: **5**  
- Backbone: **ResNet18 (ImageNet pretrained)**  
- Heads: **severity (4 classes)**, **foul type (11 classes)**  
- Device: **MPS (Mac)**  

**Best validation window: Epochs 1–3**

| Epoch | Val Loss | Severity Balanced Acc | Foul Type Balanced Acc |
|-------|----------|------------------------|-------------------------|
| **1** | **3.0287** | **0.2430**           | **0.0855**              |
| **2** | 3.1253   | 0.2913                | 0.0884                  |
| **3** | 3.2278   | 0.2864                | 0.0900                  |

The model begins to **overfit after Epoch 3**, so Epochs 1–3 represent the *true* single-view baseline that we will compare against later.

---

## ▶️ Running the Baseline Training

    python3 code/train_baseline.py \
      --data_root "/path/to/SoccerNetData/mvfouls" \
      --epochs 5 \
      --batch_size 4 \
      --num_frames 16 \
      --max_train_samples 1000 \
      --max_val_samples 300 \
      --use_pretrained

This produces:

    results/baseline_best.pth

which stores the best-performing checkpoint.

---

## 🧪 Debug / Utility Scripts

### Inspect annotation structure

    python3 code/inspect_annotations.py --root /path/to/SoccerNetData/mvfouls

### Test dataset loading

    python3 code/test_dataset.py

### Test model output shapes

    python3 code/test_model_shapes.py

---

## 🌐 Final Project Website

The final website (in `website/`) will contain:

- Proposal  
- Midterm report  
- Baseline model & results  
- Multi-view improvements  
- Architecture diagrams  
- Result visualizations  
- Final discussion + future work  
- Downloadable code & checkpoints  

> **Important:** The webpage must not be modified after the official Canvas due date.

---

## ✔ README Complete

This README reflects the current state of the repository, the completed single-view baseline, and how to reproduce the main experiments.