# CS566 — VAR Foul Recognition (SoccerNet MV-Foul)

This repository contains our final project for **CS566: Computer Vision**, where we implement and experiment with the **SoccerNet Multi-View Foul Recognition** task.  
Our work includes the baseline MVNetwork implementation, dataset preprocessing, training pipeline, evaluation scripts, and a final project website summarizing results.

---

## 📁 Repository Structure

    CS566-VAR-Foul-Project/
    │
    ├── code/                # Model, dataset, training, evaluation code
    │   ├── mvfoul_dataset.py
    │   ├── train_baseline.py
    │   ├── utils.py
    │   └── ...
    │
    ├── scripts/             # Utility scripts (non-training)
    │   ├── download_train.py
    │   ├── download_valid.py
    │   └── ...
    │
    ├── results/             # Logs, metrics, confusion matrices, saved models
    │
    ├── figures/             # Plots, visualizations, and sample frame outputs
    │
    ├── website/             # Final project webpage (index.html + assets)
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

The goal of this project is to perform **foul recognition in soccer clips** using the **SoccerNet MV-Foul dataset**, which provides synchronized multi-view video of match events along with annotations.

We aim to:

1. **Implement the official MVNetwork baseline** for predicting:
   - Foul severity  
   - Foul type  

2. **Explore improvements**, including:
   - Alternative video backbones  
   - Different multi-view fusion strategies  
   - Optimization & hyperparameter tuning  

3. **Evaluate performance** using balanced accuracy and per-class metrics.

4. **Publish a project webpage** summarizing methodology, results, visualizations, and learnings.

---

## 📦 Downloading the SoccerNet MV-Foul Dataset

To access the dataset:

### 1️⃣ Sign the SoccerNet NDA

Visit: https://www.soccer-net.org/data  
Once approved, you will receive the video password.

### 2️⃣ Install the SoccerNet API

    pip install SoccerNet --upgrade

### 3️⃣ Download the dataset splits

We provide utility scripts under `scripts/`.

Download the **train** split:

    python3 scripts/download_train.py

Download the **valid** split:

    python3 scripts/download_valid.py

### 4️⃣ Set your data directory

Before running these scripts, open each file and set:

```python
data_dir = "/path/to/SoccerNetData"