import torch
import numpy as np
from train_dynamics import TrainDynamics
from track_profile import TrackProfile
from mpc_controller import SimpleMPC
from policy_network import PolicyNetwork
from data_loader import DataLoader


class SimulationCore:
    def __init__(self, total_distance=15000):
        self.default_distance = total_distance
        self.track = TrackProfile(self.default_distance)
        self.dynamics = TrainDynamics("CR400AF")
        self.mpc = SimpleMPC(self.dynamics, self.track)
        self.net = PolicyNetwork()

        self.curr_pos = 0.0
        self.curr_vel = 0.0
        self.dataset = []
        self.external_data = {"pos": [], "vel": [], "u": [], "target": []}

        self.is_metro_mode = False

    def reset(self):
        v_type = "METRO" if self.is_metro_mode else "CR400AF"
        self.dynamics = TrainDynamics(v_type)
        self.mpc.reset()
        self.curr_pos = 0.0
        self.curr_vel = 0.0

    def get_atp_limit(self, pos):
        idx = int(np.clip(pos, 0, self.track.total_dist))
        if idx >= len(self.track.static_limit): idx = -1
        return self.track.static_limit[idx]

    def _get_ai_state(self):
        if self.is_metro_mode:
            NORM_V = 25.0
            NORM_ERR = 10.0
        else:
            NORM_V = 100.0
            NORM_ERR = 20.0

        target_v = self.track.get_target_v(self.curr_pos)

        # [策略优化] 欺骗 AI：告诉它目标速度比实际低一点点 (0.5 km/h)
        # 这样 AI 会倾向于跑得慢一点，留出安全余量
        safe_target_v = target_v
        if self.is_metro_mode:
            safe_target_v = max(0, target_v - 0.5 / 3.6)

        v_err = safe_target_v - self.curr_vel
        scalar = [self.curr_vel / NORM_V, v_err / NORM_ERR, self.mpc.last_u]

        seq = []
        look_dist = 0
        v_sim = max(self.curr_vel, 1.0)
        for _ in range(10):
            look_dist += v_sim * 1.0
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
        if mode == "MPC" and self.is_metro_mode:
            self.is_metro_mode = False
            self.track = TrackProfile(self.default_distance)
            self.reset()

        if self.curr_pos >= self.track.total_dist:
            return {"finished": True}

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

        # === ATP 安全防护 (针对模式二的强力修正) ===
        atp_limit = self.get_atp_limit(self.curr_pos)
        is_emergency = False

        if self.is_metro_mode:
            # 地铁模式：零容忍策略
            # CSV里的"目标速度"其实就是ATO曲线，不能超
            if self.curr_vel > target_v:
                # 一旦超过目标速度，无视 AI，强制介入
                # 如果超得不多，用轻刹车；超多了用急刹车
                over_speed = self.curr_vel - target_v
                if over_speed > 1.0 / 3.6:
                    u = -1.0  # 严重超速 (>1km/h) -> 紧急制动
                    is_emergency = True
                else:
                    u = min(u, -0.5)  # 轻微超速 -> 常用制动
        else:
            # 高铁模式：原来的宽松策略
            if self.curr_vel > (atp_limit + 3.0 / 3.6):
                u = -1.0
                is_emergency = True
            elif self.curr_vel > atp_limit:
                u = min(u, -0.5)

        self.curr_pos, self.curr_vel, _ = self.dynamics.step(self.curr_pos, self.curr_vel, u)

        force = self.dynamics.current_force
        res = 1.0 + 0.01 * self.curr_vel + 0.0008 * self.curr_vel ** 2
        acc = (force - res) / self.dynamics.mass
        if self.curr_vel < 0.1 and 0 < force <= 3.0: acc = 0.0

        return {
            "finished": False,
            "pos": self.curr_pos,
            "vel": self.curr_vel * 3.6,
            "target_v": target_v * 3.6,
            "acc": acc,
            "atp_limit": atp_limit * 3.6,
            "is_emergency": is_emergency,
            "dataset_count": len(self.dataset)
        }