
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3PM25(nn.Module):
    """
    DINOv3 ViT-S/16 + forecasting head cho dự báo PM2.5.

    Input : (batch, 96, 7)
    Output: (batch, 24, 1)

    DINOv3 backbone được đóng băng; chỉ huấn luyện forecasting head.
    """

    def __init__(
        self,
        repo_dir="external/dinov3",
        weights_path=(
            "checkpoints/dinov3/"
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        ),
        image_size=224,
        forecast_hours=24,
        dropout=0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.forecast_hours = forecast_hours

        # Nạp DINOv3 ViT-S/16 từ repository local.
        self.backbone = torch.hub.load(
            repo_dir,
            "dinov3_vits16",
            source="local",
            weights=weights_path,
        )

        # Đóng băng backbone pretrained.
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        # ViT-S/16 tạo vector đặc trưng 384 chiều.
        self.forecasting_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_hours),
        )

    def train(self, mode=True):
        # Giữ backbone ở chế độ eval khi huấn luyện head.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        """
        x: (B, 96, 7), dữ liệu đã chuẩn hóa.
        """
        if x.ndim != 3 or x.shape[1:] != (96, 7):
            raise ValueError(
                f"Input cần có dạng (B, 96, 7), nhận được {tuple(x.shape)}"
            )

        batch_size = x.shape[0]

        # Biểu diễn 7 biến theo 96 giờ thành ảnh RGB.
        # Nội suy ảnh ở đây chỉ là cách biểu diễn thử nghiệm.
        image = x.transpose(1, 2).unsqueeze(1)  # (B, 1, 7, 96)
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, 224, 224)

        # DINOv3 yêu cầu 3 kênh ảnh.
        image = image.repeat(1, 3, 1, 1)

        # Chuỗi dữ liệu đã chuẩn hóa có thể âm,
        # vì vậy ánh xạ tuyến tính sang [0, 1] để thử nghiệm.
        image = torch.sigmoid(image)

        # Trích xuất đặc trưng; không tính gradient cho backbone.
        with torch.no_grad():
            features = self.backbone(image)

        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            else:
                raise RuntimeError(
                    "Không tìm thấy x_norm_clstoken trong đầu ra DINOv3."
                )

        if features.ndim > 2:
            features = features.flatten(start_dim=1)

        if features.shape != (batch_size, 384):
            raise RuntimeError(
                f"Đặc trưng cần có dạng (B, 384), "
                f"nhận được {tuple(features.shape)}"
            )

        # (B, 384) -> (B, 24, 1)
        prediction = self.forecasting_head(features)
        return prediction.unsqueeze(-1)

