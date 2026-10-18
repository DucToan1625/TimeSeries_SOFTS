
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def _forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        PM25:
            X: [B, 96, 7]
            Y: [B, 24, 1]

        SOFTS chỉ sử dụng x_enc để dự báo.
        Decoder và time marks được giữ để tương thích API mô hình.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if self.args.data == "PM25":
            batch_x_mark = None
            batch_y_mark = None

            # SOFTS không dùng x_dec trong forecast().
            dec_inp = torch.zeros(
                batch_x.size(0),
                self.args.pred_len,
                1,
                dtype=torch.float32,
                device=self.device
            )

            outputs = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark
            )

            # SOFTS mặc định chiếu dự báo cho từng biến đầu vào.
            # Chọn biến PM2.5 ở vị trí 0 theo thứ tự feature đã xác nhận.
            if outputs.shape[-1] == batch_x.shape[-1]:
                outputs = outputs[:, :, 0:1]

            elif outputs.shape[-1] != 1:
                raise ValueError(
                    "SOFTS phải trả về 1 hoặc 7 biến, "
                    f"nhưng nhận được shape {tuple(outputs.shape)}"
                )

            if outputs.shape[1] != self.args.pred_len:
                raise ValueError(
                    f"Mô hình cần dự báo {self.args.pred_len} bước, "
                    f"nhận được {outputs.shape[1]} bước"
                )

            if outputs.shape != batch_y.shape:
                raise ValueError(
                    f"Shape dự báo {tuple(outputs.shape)} khác "
                    f"shape nhãn {tuple(batch_y.shape)}"
                )

            return outputs, batch_y

        # Luồng gốc dành cho các dataset SOFTS khác.
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(
            batch_y[:, -self.args.pred_len:, :]
        ).float()

        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp],
            dim=1
        ).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
        else:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark
            )

        if self.args.output_attention:
            outputs = outputs[0]

        f_dim = -1 if self.args.features == "MS" else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                outputs, targets = self._forward(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark
                )

                loss = criterion(outputs, targets)
                total_loss.update(loss.item(), batch_x.size(0))

        self.model.train()
        return total_loss.avg

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = (
            torch.cuda.amp.GradScaler()
            if self.args.use_amp
            else None
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_start = time.time()
            train_loss_sum = 0.0
            train_sample_count = 0

            for i, (
                batch_x, batch_y, batch_x_mark, batch_y_mark
            ) in enumerate(train_loader):

                model_optim.zero_grad(set_to_none=True)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, targets = self._forward(
                            batch_x, batch_y,
                            batch_x_mark, batch_y_mark
                        )
                        loss = criterion(outputs, targets)

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()

                else:
                    outputs, targets = self._forward(
                        batch_x, batch_y,
                        batch_x_mark, batch_y_mark
                    )
                    loss = criterion(outputs, targets)
                    loss.backward()
                    model_optim.step()

                batch_size = batch_x.size(0)
                train_loss_sum += loss.item() * batch_size
                train_sample_count += batch_size

                if (i + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}, "
                        f"Iteration {i + 1}/{train_steps}, "
                        f"Loss: {loss.item():.7f}"
                    )

            train_loss = train_loss_sum / max(train_sample_count, 1)

            vali_loss = self.vali(
                vali_data, vali_loader, criterion
            )

            # Giữ hành vi đánh giá test theo từng epoch của code gốc.
            test_loss = self.vali(
                test_data, test_loader, criterion
            )

            print(
                f"Epoch: {epoch + 1}, "
                f"Time: {time.time() - epoch_start:.2f}s | "
                f"Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} "
                f"Test Loss: {test_loss:.7f}"
            )

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(
                model_optim, epoch + 1, self.args
            )

        best_model_path = os.path.join(path, "checkpoint.pth")

        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint tốt nhất: {best_model_path}"
            )

        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

        if not self.args.save_model:
            import shutil
            shutil.rmtree(path, ignore_errors=True)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            checkpoint_path = os.path.join(
                self.args.checkpoints,
                setting,
                "checkpoint.pth"
            )

            self.model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=self.device
                )
            )

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()

        mse = AverageMeter()
        mae = AverageMeter()

        self.model.eval()

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                outputs, targets = self._forward(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark
                )

                batch_size = batch_x.size(0)

                mse.update(
                    mse_loss(outputs, targets).item(),
                    batch_size
                )
                mae.update(
                    mae_loss(outputs, targets).item(),
                    batch_size
                )

                all_outputs.append(outputs.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

        print(f"Test MSE (standardized): {mse.avg:.6f}")
        print(f"Test MAE (standardized): {mae.avg:.6f}")

        # Lưu dự báo và nhãn đã chuẩn hóa để có thể đánh giá tiếp.
        outputs_array = np.concatenate(all_outputs, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)

        result_path = os.path.join(
            self.args.checkpoints,
            setting
        )
        os.makedirs(result_path, exist_ok=True)

        np.savez(
            os.path.join(result_path, "pm25_test_predictions.npz"),
            predictions=outputs_array,
            targets=targets_array
        )

        return
