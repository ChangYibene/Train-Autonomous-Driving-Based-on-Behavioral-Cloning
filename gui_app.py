import sys
import traceback
import numpy as np  # <--- 已补上此引用
import torch
import torch.optim as optim
import torch.nn as nn
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QMessageBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QApplication, QHeaderView)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# === 导入自定义模块 ===
try:
    from train_dynamics import TrainDynamics
    from track_profile import TrackProfile
    from mpc_controller import SimpleMPC
    from policy_network import PolicyNetwork
except ImportError as e:
    print(f"【严重错误】缺少必要模块: {e}")
    sys.exit(1)


# ==========================================
# 1. 后台训练线程类
# ==========================================
class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int, int, float)
    finished_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, net, dataset, epochs=500, lr=0.002):
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        self._is_running = True

    def run(self):
        try:
            if not self.dataset:
                raise ValueError("数据集为空！请先运行 MPC 采集数据。")

            # 数据预处理
            try:
                # dataset结构: [(([scalar], [seq]), action), ...]
                scalar_data = [d[0][0] for d in self.dataset]
                seq_data = [d[0][1] for d in self.dataset]
                actions = [[d[1]] for d in self.dataset]

                X_scalar = torch.tensor(scalar_data, dtype=torch.float32)
                # [N, 10] -> [N, 1, 10] 用于 Conv1d
                X_seq = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(1)
                Y = torch.tensor(actions, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"数据格式错误，这通常是因为 MPC 代码版本过旧导致的。\n详情: {e}")

            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            self.net.train()
            final_loss = 0.0

            for e in range(self.epochs):
                if not self._is_running: break
                optimizer.zero_grad()

                # 双输入前向传播
                try:
                    output = self.net(X_scalar, X_seq)
                except TypeError:
                    raise RuntimeError("PolicyNetwork 版本过旧！请更新 policy_network.py 以支持双输入。")

                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

                if e % 10 == 0 or e == self.epochs - 1:
                    self.progress_signal.emit(e + 1, self.epochs, final_loss)

            self.finished_signal.emit(final_loss)

        except Exception as e:
            self.error_signal.emit(traceback.format_exc())

    def stop(self):
        self._is_running = False


# ==========================================
# 2. 主界面类
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CR400AF 自动驾驶仿真 (最终修复版)")
        self.resize(1280, 800)

        # 1. 模块初始化
        self.total_distance = 15000
        self.track = TrackProfile(self.total_distance)
        self.dynamics = TrainDynamics()
        self.mpc = SimpleMPC(self.dynamics, self.track)
        self.net = PolicyNetwork()

        # 检查 MPC 版本兼容性
        if not hasattr(self.mpc, 'last_u'):
            QMessageBox.critical(self, "版本不匹配", "mpc_controller.py 版本过旧！\n请添加 self.last_u 属性。")

        self.dataset = []
        self.sim_mode = "IDLE"
        self.sim_data = {"pos": [], "vel": [], "acc": [], "target": []}
        self.train_thread = None

        self.setup_ui()
        self.generate_default_curve()

        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)

    def update_plot(self):
        """实时更新仿真曲线"""
        self.ax.clear()
        self.ax.plot(self.track.x, self.track.target_curve * 3.6, 'r-', alpha=0.4, label='Target')
        if self.sim_data["pos"]:
            self.ax.plot(self.sim_data["pos"], self.sim_data["vel"], 'b-', linewidth=2, label='Actual Speed')

        self.ax.set_ylim(0, 400)
        self.ax.set_ylabel("Speed (km/h)")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.canvas.draw()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧面板
        left_panel = QWidget()
        l_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(380)

        # 线路表格
        grp_line = QGroupBox("1. 线路配置")
        v_line = QVBoxLayout(grp_line)
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["起点", "终点", "限速"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        btn_gen = QPushButton("应用线路设置")
        btn_gen.clicked.connect(self.generate_curve_from_table)
        v_line.addWidget(self.table)
        v_line.addWidget(btn_gen)
        l_layout.addWidget(grp_line)

        # 控制按钮
        grp_ctrl = QGroupBox("2. 操作面板")
        v_ctrl = QVBoxLayout(grp_ctrl)
        self.btn_mpc = QPushButton("启动 MPC (采集)")
        self.btn_mpc.clicked.connect(lambda: self.start_sim("MPC"))
        self.btn_train = QPushButton("训练模型 (多线程)")
        self.btn_train.clicked.connect(self.start_training_thread)
        self.btn_ai = QPushButton("启动 AI (验证)")
        self.btn_ai.clicked.connect(lambda: self.start_sim("AI"))
        v_ctrl.addWidget(self.btn_mpc)
        v_ctrl.addWidget(self.btn_train)
        v_ctrl.addWidget(self.btn_ai)
        l_layout.addWidget(grp_ctrl)

        # 仪表盘
        grp_dash = QGroupBox("3. 状态监视")
        v_dash = QVBoxLayout(grp_dash)
        self.lbl_status = QLabel("就绪")
        self.lbl_val = QLabel("位置: 0.0m | 速度: 0.0km/h")
        self.lbl_acc = QLabel("加速度: 0.00 m/s²")
        self.lbl_warn = QLabel("")
        self.lbl_warn.setStyleSheet("color: red; font-weight: bold;")
        v_dash.addWidget(self.lbl_status)
        v_dash.addWidget(self.lbl_val)
        v_dash.addWidget(self.lbl_acc)
        v_dash.addWidget(self.lbl_warn)
        l_layout.addWidget(grp_dash)

        l_layout.addStretch()

        # 右侧绘图
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        r_layout.addWidget(self.canvas)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

    def get_state_context(self, current_pos, current_vel, last_u):
        """构建双模态状态"""
        target_v = self.track.get_target_v(current_pos)
        v_err = target_v - current_vel
        scalar = [current_vel / 100.0, v_err / 20.0, last_u]

        seq = []
        look_dist = 0
        v_sim = max(current_vel, 1.0)
        for _ in range(10):
            look_dist += v_sim * 1.0
            p_next = current_pos + look_dist
            t_v_next = self.track.get_target_v(p_next)
            seq.append((t_v_next - current_vel) / 20.0)
        return scalar, seq

    def generate_default_curve(self):
        defaults = [(0, 5000, 200), (5000, 10000, 350), (10000, 12000, 160), (12000, 15000, 80)]
        for r, (s, e, v) in enumerate(defaults):
            self.table.setItem(r, 0, QTableWidgetItem(str(s)))
            self.table.setItem(r, 1, QTableWidgetItem(str(e)))
            self.table.setItem(r, 2, QTableWidgetItem(str(v)))
        self.generate_curve_from_table()

    def generate_curve_from_table(self):
        try:
            self.track.reset_limits()
            for r in range(self.table.rowCount()):
                s = self.table.item(r, 0)
                e = self.table.item(r, 1)
                v = self.table.item(r, 2)
                if s and e and v:
                    self.track.apply_limit(float(s.text()), float(e.text()), float(v.text()))
            self.track.calculate_braking_curve()
            self.update_plot()
        except ValueError:
            QMessageBox.warning(self, "错误", "输入数据无效")

    def start_sim(self, mode):
        if mode == "AI" and not self.dataset:
            QMessageBox.warning(self, "警告", "无训练数据，请先运行 MPC 并训练！")
            return

        self.sim_mode = mode
        self.dynamics = TrainDynamics()
        self.mpc.reset()  # 只有 MPC 模式下 reset 会清空 last_u
        self.curr_pos = 0.0
        self.curr_vel = 0.0
        self.sim_data = {"pos": [], "vel": [], "acc": [], "target": []}

        if mode == "MPC":
            self.dataset = []
            self.lbl_status.setText("状态: MPC 采集数据中...")
        else:
            self.lbl_status.setText("状态: AI 验证中...")

        self.timer.start(100)

    def game_loop(self):
        try:
            # 1. 结束判断
            if self.curr_pos >= self.total_distance:
                self.timer.stop()
                self.lbl_status.setText("运行结束")
                if self.sim_mode == "MPC":
                    QMessageBox.information(self, "完成", f"采集数据: {len(self.dataset)} 条")
                return

            # 2. 获取状态
            scalar, seq = self.get_state_context(self.curr_pos, self.curr_vel, self.mpc.last_u)
            # 【修复点】这里需要 numpy 来处理索引，现在已经 import 了
            atp_limit = self.track.static_limit[int(np.clip(self.curr_pos, 0, self.total_distance))]

            # 3. 控制决策
            u = 0.0
            if self.sim_mode == "MPC":
                u = self.mpc.get_action(self.curr_pos, self.curr_vel)
                self.dataset.append(((scalar, seq), u))

            elif self.sim_mode == "AI":
                t_scalar = torch.tensor([scalar], dtype=torch.float32)
                t_seq = torch.tensor([[seq]], dtype=torch.float32)  # [1, 1, 10]
                with torch.no_grad():
                    u = self.net(t_scalar, t_seq).item()

            # 4. ATP 安全防护 (AI 模式下的兜底)
            is_emergency = False
            if self.curr_vel > (atp_limit + 1.0 / 3.6):  # 超速 1km/h
                u = -1.0
                is_emergency = True
            elif self.curr_vel > atp_limit:
                u = min(u, -0.5)

            # 5. 物理更新
            self.curr_pos, self.curr_vel, _ = self.dynamics.step(self.curr_pos, self.curr_vel, u)

            # 6. 计算显示用加速度
            force = self.dynamics.current_force
            res = 1.0 + 0.01 * self.curr_vel + 0.0008 * self.curr_vel ** 2
            acc = 0.0
            if not (self.curr_vel < 0.1 and 0 < force <= 3.0):
                acc = (force - res) / self.dynamics.mass

            # 7. 更新UI数据
            self.sim_data["pos"].append(self.curr_pos)
            self.sim_data["vel"].append(self.curr_vel * 3.6)
            self.sim_data["acc"].append(acc)

            if len(self.sim_data["pos"]) % 2 == 0:
                self.lbl_val.setText(f"位置: {self.curr_pos:.1f}m | 速度: {self.curr_vel * 3.6:.1f} km/h")
                self.lbl_acc.setText(f"加速度: {acc:.2f} m/s²")
                self.lbl_warn.setText("⚠️ ATP 触发!" if is_emergency else "")

            if len(self.sim_data["pos"]) % 5 == 0:
                self.update_plot()

        except Exception as e:
            self.timer.stop()
            print(traceback.format_exc())
            QMessageBox.critical(self, "运行错误", f"发生未捕获异常:\n{e}")

    # --- 训练逻辑 ---
    def start_training_thread(self):
        if not self.dataset:
            QMessageBox.warning(self, "提示", "请先运行 MPC 采集数据")
            return

        self.btn_mpc.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_ai.setEnabled(False)
        self.lbl_status.setText("训练初始化...")

        self.train_thread = TrainingWorker(self.net, self.dataset)
        self.train_thread.progress_signal.connect(
            lambda c, t, l: self.lbl_status.setText(f"训练中: {c}/{t} Loss:{l:.4f}"))
        self.train_thread.finished_signal.connect(self.on_train_finished)
        self.train_thread.error_signal.connect(self.on_train_error)
        self.train_thread.start()

    def on_train_finished(self, loss):
        self.lbl_status.setText(f"训练完成 Loss: {loss:.5f}")
        QMessageBox.information(self, "成功", "模型已更新")
        self.btn_mpc.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_ai.setEnabled(True)

    def on_train_error(self, msg):
        self.lbl_status.setText("训练失败")
        self.btn_mpc.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_ai.setEnabled(True)
        QMessageBox.critical(self, "训练错误", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())