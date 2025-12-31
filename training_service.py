import traceback
import torch
import torch.optim as optim
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingWorker(QThread):
    # 标准化的信号接口
    progress = pyqtSignal(int, int, float) # current, total, loss
    finished = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, net, dataset, epochs=500):
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.epochs = epochs
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            if not self.dataset:
                raise ValueError("Dataset is empty.")

            # 数据转换 (CPU 操作)
            scalars = [d[0][0] for d in self.dataset]
            seqs = [d[0][1] for d in self.dataset]
            actions = [[d[1]] for d in self.dataset]

            X_scalar = torch.tensor(scalars, dtype=torch.float32)
            X_seq = torch.tensor(seqs, dtype=torch.float32).unsqueeze(1)
            Y = torch.tensor(actions, dtype=torch.float32)

            optimizer = optim.Adam(self.net.parameters(), lr=0.002)
            criterion = nn.MSELoss()
            self.net.train()

            final_loss = 0.0
            for e in range(self.epochs):
                if not self._running: break

                optimizer.zero_grad()
                output = self.net(X_scalar, X_seq)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()

                final_loss = loss.item()
                if e % 10 == 0:
                    self.progress.emit(e+1, self.epochs, final_loss)

            self.finished.emit(final_loss)

        except Exception:
            self.error.emit(traceback.format_exc())