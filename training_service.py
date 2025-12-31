import traceback
import torch
import torch.optim as optim
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    # 标准化的信号接口
    progress_signal = pyqtSignal(int, int, float)  # current, total, loss
    finished_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, net, dataset, epochs=500, lr=0.002):
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
            if not self.dataset:
                raise ValueError("训练数据集为空！请先采集数据或读取文件。")

            # === 1. 数据解包与预处理 ===
            try:
                # dataset 结构: [ (([scalar...], [seq...]), action), ... ]

                # 提取标量部分 [N, 3]
                scalar_data = [d[0][0] for d in self.dataset]
                X_scalar = torch.tensor(scalar_data, dtype=torch.float32)

                # 提取序列部分 [N, 10]
                seq_data = [d[0][1] for d in self.dataset]
                X_seq = torch.tensor(seq_data, dtype=torch.float32)

                # --- 【核心修复】维度安全检查 ---
                # PolicyNetwork 的 Conv1d 需要输入形状: [Batch, Channel, Length] -> [N, 1, 10]

                # 情况A: 如果当前是 [N, 10]，说明缺少 Channel 维 -> 需要 unsqueeze(1)
                if X_seq.dim() == 2:
                    X_seq = X_seq.unsqueeze(1)

                # 情况B: 如果当前是 [N, 1, 10]，说明已经是正确的 -> 不需要动
                elif X_seq.dim() == 3 and X_seq.size(1) == 1:
                    pass

                # 情况C: 如果是 [N, 1, 1, 10] (之前报错的原因)，说明多了一维 -> 需要 squeeze
                elif X_seq.dim() == 4:
                    X_seq = X_seq.squeeze(1)  # 压缩掉多余维度

                # 最终检查
                if X_seq.dim() != 3:
                    raise ValueError(f"序列数据维度异常: {X_seq.shape}, 期望 (Batch, 1, 10)")

                # 提取标签 [N, 1]
                actions = [[d[1]] for d in self.dataset]
                Y = torch.tensor(actions, dtype=torch.float32)

            except Exception as e:
                raise ValueError(f"数据格式转换失败。\n原因: {str(e)}\n请检查 data_loader.py 生成的数据结构。")

            # === 2. 训练配置 ===
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            self.net.train()

            final_loss = 0.0

            # === 3. 训练循环 ===
            for e in range(self.epochs):
                if not self._is_running: break

                optimizer.zero_grad()

                # 前向传播 (确保 PolicyNetwork 支持双输入)
                output = self.net(X_scalar, X_seq)

                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()

                final_loss = loss.item()

                # 每 10 轮或最后一轮发送进度信号
                if e % 10 == 0 or e == self.epochs - 1:
                    self.progress_signal.emit(e + 1, self.epochs, final_loss)

            self.finished_signal.emit(final_loss)

        except Exception as e:
            # 捕获所有错误并发送给 GUI 显示，而不是直接闪退
            self.error_signal.emit(traceback.format_exc())