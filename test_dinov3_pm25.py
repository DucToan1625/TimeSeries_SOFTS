import torch
from models.DINOv3_PM25 import DINOv3PM25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DINOv3PM25().to(device)
model.eval()

x = torch.randn(2, 96, 7, device=device)

with torch.inference_mode():
    y = model(x)

print("Device:", device)
print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(y.shape))
print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
