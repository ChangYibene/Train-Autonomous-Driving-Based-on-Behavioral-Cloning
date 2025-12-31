import torch
import numpy as np
from train_dynamics import TrainDynamics
from track_profile import TrackProfile
from mpc_controller import SimpleMPC
from policy_network import PolicyNetwork
from data_loader import DataLoader

try:
    from pid_controller import PIDController
except:
    pass


class SimulationCore:
    def __init__(self, total_distance=15000):
        self.default_distance = total_distance
        self.track = TrackProfile(self.default_distance)
        self.dynamics = TrainDynamics("CR400AF")
        self.mpc = SimpleMPC(self.dynamics, self.track)
        self.net = PolicyNetwork()  # V2.0 (LSTM+DeepCNN)

        try:
            self.pid = PIDController()
        except:
            self.pid = None

        self.curr_pos = 0.0
        self.curr_vel = 0.0
        self.step_count = 0
        self.max_steps = 5000

        self.dataset = []
        self.external_data = {"pos": [], "vel": [], "u": [], "target": []}
        self.is_metro_mode = False

    def reset(self):
        v_type = "METRO" if self.is_metro_mode else "CR400AF"
        self.dynamics = TrainDynamics(v_type)
        self.mpc.reset()
        if self.pid: self.pid.reset()
        self.curr_pos = 0.0
        self.curr_vel = 0.0
        self.step_count = 0

    def get_atp_limit(self, pos):
        idx = int(np.clip(pos, 0, self.track.total_dist))
        if idx >= len(self.track.static_limit): idx = -1
        return self.track.static_limit[idx]

    def _get_ai_state(self):
        # [必须同步] 视野参数
        LOOK_AHEAD_STEPS = 50

        if self.is_metro_mode:
            NORM_V = 25.0
            NORM_ERR = 10.0
        else:
            NORM_V = 100.0
            NORM_ERR = 20.0

        target_v = self.track.get_target_v(self.curr_pos)

        # 欺骗策略
        safe_target_v = target_v
        if self.is_metro_mode:
            safe_target_v = max(0, target_v - 0.5 / 3.6)

        v_err = safe_target_v - self.curr_vel
        scalar = [self.curr_vel / NORM_V, v_err / NORM_ERR, self.mpc.last_u]

        # 序列构建 (50步)
        seq = []
        look_dist = 0
        v_sim = max(self.curr_vel, 1.0)

        for _ in range(LOOK_AHEAD_STEPS):
            look_dist += v_sim * 0.2
            p_next = self.curr_pos + look_dist
            t_v_next = self.track.get_target_v(p_next)
            seq.append((t_v_next - self.curr_vel) / NORM_ERR)

        return scalar, seq

    def load_external_data(self, folder_path):
        dataset, ext_data, track, msg = DataLoader.load_yizhuang_data(folder_path)
        if dataset is None: return False, msg
        self.dataset = dataset
        self.external_data = ext_data
        self.track = track
        self.is_metro_mode = True
        return True, msg

    def step(self, mode):
        # 模式保护
        if mode == "MPC" and self.is_metro_mode:
            self.is_metro_mode = False
            self.track = TrackProfile(self.default_distance)
            self.reset()

        self.step_count += 1

        # 终止条件
        status = "RUNNING"
        dist_err = self.track.total_dist - self.curr_pos
        if dist_err < -5.0:
            status = "OVERRUN"
        elif abs(dist_err) < 2.0 and self.curr_vel < (0.2 / 3.6):
            status = "SUCCESS"
        elif dist_err > 10.0 and self.curr_vel < 0.01 and self.step_count > 50:
            status = "STALL"
        elif self.step_count > self.max_steps:
            status = "TIMEOUT"
        elif self.is_metro_mode and self.curr_pos >= self.track.total_dist:
            status = "DATA_END"

        if status != "RUNNING":
            return {"status": status, "pos": self.curr_pos, "vel": self.curr_vel * 3.6, "acc": 0.0,
                    "target_v": 0.0, "atp_limit": 0.0, "is_emergency": False, "dataset_count": len(self.dataset)}

        # 状态获取
        scalar, seq = self._get_ai_state()
        target_v = self.track.get_target_v(self.curr_pos)
        u = 0.0

        if mode == "MPC":
            u = self.mpc.get_action(self.curr_pos, self.curr_vel)
            self.dataset.append(((scalar, seq), u))

        elif mode == "AI":
            t_scalar = torch.tensor([scalar], dtype=torch.float32)
            t_seq = torch.tensor([seq], dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                u = self.net(t_scalar, t_seq).item()

        elif mode == "PID":
            if self.pid: u = self.pid.get_action(self.curr_vel, target_v)

        # ATP 防护
        atp_limit = self.get_atp_limit(self.curr_pos)
        is_emergency = False

        if self.is_metro_mode:
            if self.curr_vel > target_v:
                over_speed = self.curr_vel - target_v
                if over_speed > 1.0 / 3.6:
                    u = -1.0
                    is_emergency = True
                else:
                    u = min(u, -0.5)
        else:
            if self.curr_vel > (atp_limit + 3.0 / 3.6):
                u = -1.0
                is_emergency = True
            elif self.curr_vel > atp_limit:
                u = min(u, -0.5)

        # 物理更新
        self.curr_pos, self.curr_vel, _ = self.dynamics.step(self.curr_pos, self.curr_vel, u)
        self.mpc.last_u = u

        force = self.dynamics.current_force
        res = 1.0 + 0.01 * self.curr_vel + 0.0008 * self.curr_vel ** 2
        acc = (force - res) / self.dynamics.mass
        if self.curr_vel < 0.1 and 0 < force <= 3.0: acc = 0.0

        return {
            "status": "RUNNING",
            "pos": self.curr_pos,
            "vel": self.curr_vel * 3.6,
            "target_v": target_v * 3.6,
            "acc": acc,
            "atp_limit": atp_limit * 3.6,
            "is_emergency": is_emergency,
            "dataset_count": len(self.dataset)
        }