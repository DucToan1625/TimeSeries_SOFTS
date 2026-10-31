import torch
import torch.nn as nn

class DinoV3Encoder(nn.Module):
    def __init__(self):
        super(DinoV3Encoder, self).__init__()
        # Khởi tạo mô hình DINOv3 backbone
        # (Đảm bảo code gọi torch.hub hoặc class tương ứng của bạn ở đây)
        pass

    def forward(self, x):
        return x


if __name__ == "__main__":
    print("Đang kiểm tra file Dino_V3.py...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Tạo tensor giả lập đầu vào (Ví dụ: Batch=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    model = DinoV3Encoder().to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
    print("Test thành công! Output shape:", output.shape)
