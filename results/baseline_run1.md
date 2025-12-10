# Baseline (Single-View, ResNet18, Pretrained)

- Train samples: 1000
- Val samples: 300
- Frames per clip: 16
- Batch size: 4
- Epochs: 5
- Backbone: ResNet18 (ImageNet pretrained)
- Heads: severity (4 classes), foul type (11 classes)
- Device: MPS (Mac)

Best-ish validation region: epochs 1–3

Epoch 1:
- val_loss = 3.0287
- bal_acc_sev = 0.2430
- bal_acc_type = 0.0855

Epoch 2:
- val_loss = 3.1253
- bal_acc_sev = 0.2913
- bal_acc_type = 0.0884

Epoch 3:
- val_loss = 3.2278
- bal_acc_sev = 0.2864
- bal_acc_type = 0.0900