# Multiview Baseline (2 views, ResNet18, Pretrained)

- Train samples: 1000  
- Val samples: 300  
- Frames per view: 16  
- Views per action: 2  
- Batch size: 2  
- Epochs: 3  
- Device: MPS (Mac, PyTorch 2.x)  

Epoch 1:
- val_loss = 3.2196  
- sev_loss = 1.3326  
- type_loss = 1.8869  
- bal_acc_sev = 0.2383  
- bal_acc_type = 0.0810  

Epoch 2:
- val_loss = 3.1427  
- sev_loss = 1.2661  
- type_loss = 1.8766  
- bal_acc_sev = 0.2737  
- bal_acc_type = 0.0928  

Epoch 3:
- val_loss = 2.9233  
- sev_loss = 1.2649  
- type_loss = 1.6584  
- bal_acc_sev = 0.2548  
- bal_acc_type = 0.1091  