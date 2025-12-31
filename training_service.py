import traceback
import torch
import torch.optim as optim
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal


class PhysicsAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 改用 SmoothL1Loss，比 MSE 更稳健
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_u, target_u, v_err_norm):
        basic_loss = self.criterion(pred_u, target_u)

        # 物理惩罚: 超速(v_err<0) 且 刹车不足(pred > target)
        is_overspeed = (v_err_norm < 0).float()
        is_dangerous = (pred_u > target_u).float()

        # 惩罚系数
        weights = 1.0 + 5.0 * is_overspeed * is_dangerous

        return torch.mean(basic_loss * weights)


class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int, int, float)
    finished_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, net, dataset, epochs=1000, lr=0.0005):  # 降低 LR，增加 Epoch
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            if not self.dataset: raise ValueError("无数据")

            # 解包数据
            scalar_data = [d[0][0] for d in self.dataset]
            seq_data = [d[0][1] for d in self.dataset]
            actions = [[d[1]] for d in self.dataset]

            X_scalar = torch.tensor(scalar_data, dtype=torch.float32)
            X_seq = torch.tensor(seq_data, dtype=torch.float32)
            Y = torch.tensor(actions, dtype=torch.float32)

            if X_seq.dim() == 2: X_seq = X_seq.unsqueeze(1)  # [N, 50] -> [N, 1, 50]

            V_err = X_scalar[:, 1:2]

            # 配置
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
            criterion = PhysicsAwareLoss()
            self.net.train()

            # Batch 训练
            batch_size = 128  # 加大 Batch Size
            num_samples = len(Y)
            final_loss = 0.0

            for e in range(self.epochs):
                if not self._is_running: break

                perm = torch.randperm(num_samples)
                epoch_loss = 0.0
                batch_cnt = 0

                for i in range(0, num_samples, batch_size):
                    idx = perm[i: i + batch_size]

                    optimizer.zero_grad()
                    pred = self.net(X_scalar[idx], X_seq[idx])
                    loss = criterion(pred, Y[idx], V_err[idx])
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_cnt += 1

                final_loss = epoch_loss / batch_cnt

                if e % 20 == 0 or e == self.epochs - 1:
                    self.progress_signal.emit(e + 1, self.epochs, final_loss)

            self.finished_signal.emit(final_loss)

        except Exception as e:
            self.error_signal.emit(traceback.format_exc())