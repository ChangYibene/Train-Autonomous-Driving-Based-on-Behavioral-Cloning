import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Deep CNN + LSTM 策略网络 (V2.0)
    适配 50 步长长序列输入，增强远距离特征提取能力
    """

    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # 1. 视觉感知 (处理 50 步长的序列)
        # 输入: [Batch, 1, 50]
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> [Batch, 16, 25]

            # Layer 2
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> [Batch, 32, 12]

            # Layer 3
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [Batch, 64, 1]
        )

        # 2. 记忆感知 (LSTM)
        # 输入: [Batch, 1, 3] (v, err, last_u)
        self.lstm = nn.LSTM(input_size=3, hidden_size=128, num_layers=2, batch_first=True)

        # 3. 决策层
        self.fc = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, scalar_state, seq_state):
        # A. CNN
        # seq: [B, 1, 50] -> [B, 64]
        cnn_feat = self.cnn(seq_state).view(seq_state.size(0), -1)

        # B. LSTM
        # scalar: [B, 3] -> [B, 1, 3]
        lstm_input = scalar_state.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_feat = lstm_out[:, -1, :]  # -> [B, 128]

        # C. Fusion
        combined = torch.cat((cnn_feat, lstm_feat), dim=1)
        action = self.fc(combined)

        return action