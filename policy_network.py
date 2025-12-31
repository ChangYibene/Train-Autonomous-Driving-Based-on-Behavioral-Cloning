import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块
    用于自动提取序列中重要的路况特征
    """

    def __init__(self, channel, reduction=4):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class PolicyNetwork(nn.Module):
    """
    基于状态感知与注意力机制的策略网络 (双输入)
    Input 1: 标量状态 (当前速度, 误差, 上一步动作)
    Input 2: 序列状态 (未来10步的速度差)
    """

    def __init__(self, look_ahead_steps=10):
        super(PolicyNetwork, self).__init__()

        # 1. 标量处理分支
        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )

        # 2. 序列处理分支 (CNN + Attention)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.attention = SEAttention(channel=16)
        self.conv_flatten = nn.Linear(16 * look_ahead_steps, 64)

        # 3. 决策头
        self.fc_final = nn.Sequential(
            nn.Linear(32 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出归一化到 [-1, 1]
        )

    def forward(self, scalar_input, sequence_input):
        """
        scalar_input: [batch, 3]
        sequence_input: [batch, 1, 10]
        """
        # 分支1：标量
        scalar_feat = self.scalar_encoder(scalar_input)

        # 分支2：序列
        seq_feat = F.relu(self.conv1(sequence_input))  # -> [B, 16, 10]
        seq_feat = self.attention(seq_feat)  # -> [B, 16, 10]
        seq_feat = seq_feat.view(seq_feat.size(0), -1)  # Flatten
        seq_feat = self.conv_flatten(seq_feat)  # -> [B, 64]

        # 融合
        combined = torch.cat((scalar_feat, seq_feat), dim=1)

        # 输出
        action = self.fc_final(combined)
        return action