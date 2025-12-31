import numpy as np
import torch
from train_dynamics import TrainDynamics
from track_profile import TrackProfile
from mpc_controller import SimpleMPC
from policy_network import PolicyNetwork


class SimulationCore:
    """
    仿真核心管理器
    职责：
    1. 管理所有子模块 (物理, 线路, 控制器)
    2. 执行单步仿真 (Step)
    3. 执行 ATP 安全防护
    4. 格式化 AI 输入数据
    """

    def __init__(self, total_distance=15000):
        self.total_distance = total_distance

        # 初始化子模块
        self.track = TrackProfile(self.total_distance)
        self.dynamics = TrainDynamics()
        self.mpc = SimpleMPC(self.dynamics, self.track)
        self.net = PolicyNetwork()

        # 运行时状态
        self.curr_pos = 0.0
        self.curr_vel = 0.0
        self.dataset = []  # 存储 (state, action)

    def reset(self):
        self.dynamics = TrainDynamics()  # 重置物理状态
        self.mpc.reset()
        self.curr_pos = 0.0
        self.curr_vel = 0.0


    def get_atp_limit(self, pos):
        """获取当前位置的 ATP 硬限速"""
        idx = int(np.clip(pos, 0, self.track.total_dist))
        return self.track.static_limit[idx]

    def _get_ai_state(self):
        """构建符合论文要求的双模态状态 (Scalar + Sequence)"""
        # 1. 标量状态
        target_v = self.track.get_target_v(self.curr_pos)
        v_err = target_v - self.curr_vel
        scalar = [self.curr_vel / 100.0, v_err / 20.0, self.mpc.last_u]

        # 2. 序列状态 (前瞻10步)
        seq = []
        look_dist = 0
        v_sim = max(self.curr_vel, 1.0)
        for _ in range(10):
            look_dist += v_sim * 1.0
            p_next = self.curr_pos + look_dist
            t_v_next = self.track.get_target_v(p_next)
            seq.append((t_v_next - self.curr_vel) / 20.0)

        return scalar, seq

    def step(self, mode):
        """
        执行一步仿真
        返回: 包含所有显示所需数据的字典
        """
        if self.curr_pos >= self.total_distance:
            return {"finished": True}

        # 1. 获取状态
        scalar, seq = self._get_ai_state()
        target_v = self.track.get_target_v(self.curr_pos)

        # 2. 计算控制量 u
        u = 0.0
        if mode == "MPC":
            u = self.mpc.get_action(self.curr_pos, self.curr_vel)
            # 收集数据
            self.dataset.append(((scalar, seq), u))

        elif mode == "AI":
            t_scalar = torch.tensor([scalar], dtype=torch.float32)
            t_seq = torch.tensor([[seq]], dtype=torch.float32).unsqueeze(1)  # [1, 1, 10]
            with torch.no_grad():
                u = self.net(t_scalar, t_seq).item()

        # 3. ATP 安全防护 (独立于 AI/MPC)
        atp_limit = self.get_atp_limit(self.curr_pos)
        is_emergency = False

        if self.curr_vel > (atp_limit + 1.0 / 3.6):  # 超速 1km/h
            u = -1.0  # 紧急制动
            is_emergency = True
        elif self.curr_vel > atp_limit:
            u = min(u, -0.5)  # 常用制动

        # 4. 物理更新
        self.curr_pos, self.curr_vel, _ = self.dynamics.step(self.curr_pos, self.curr_vel, u)

        # 5. 计算高精度加速度 (用于显示)
        force = self.dynamics.current_force
        res = 1.0 + 0.01 * self.curr_vel + 0.0008 * self.curr_vel ** 2
        acc = (force - res) / self.dynamics.mass
        if self.curr_vel < 0.1 and 0 < force <= 3.0: acc = 0.0

        # 返回数据包给 GUI
        return {
            "finished": False,
            "pos": self.curr_pos,
            "vel": self.curr_vel * 3.6,  # km/h
            "target_v": target_v * 3.6,
            "acc": acc,
            "atp_limit": atp_limit * 3.6,
            "is_emergency": is_emergency,
            "dataset_count": len(self.dataset)
        }