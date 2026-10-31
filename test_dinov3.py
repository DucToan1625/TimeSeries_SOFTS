import os
import torch

REPO_DIR = os.path.abspath("external/dinov3")
WEIGHTS_PATH = os.path.abspath(
    "checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Checkpoint exists:", os.path.isfile(WEIGHTS_PATH))

model = torch.hub.load(
    REPO_DIR,
    "dinov3_vits16",
    source="local",
    weights=WEIGHTS_PATH,
)

model.eval()
model.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Device:", device)
print("DINOv3 loaded successfully!")

# Kiểm tra đầu ra với một ảnh RGB giả lập
x = torch.randn(1, 3, 224, 224, device=device)

with torch.inference_mode():
    output = model(x)

if isinstance(output, dict):
    print("Output keys:", list(output.keys()))
    for key, value in output.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape))
else:
    print("Output shape:", tuple(output.shape))
