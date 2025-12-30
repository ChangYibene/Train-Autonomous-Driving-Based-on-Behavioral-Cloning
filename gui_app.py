import sys
import torch
import torch.optim as optim
import torch.nn as nn
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QMessageBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QApplication, QHeaderView)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# === 导入自定义模块 ===
# 确保 train_dynamics.py, track_profile.py 等文件在同一目录下
from train_dynamics import TrainDynamics
from track_profile import TrackProfile
from mpc_controller import SimpleMPC
from policy_network import PolicyNetwork


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CR400AF 高速列车自动驾驶仿真平台 (模块化最终版)")
        self.resize(1280, 800)

        # 1. 初始化各模块对象
        self.total_distance = 15000  # 15公里长线路
        self.track = TrackProfile(self.total_distance)
        self.dynamics = TrainDynamics()
        self.mpc = SimpleMPC(self.dynamics, self.track)
        self.net = PolicyNetwork()

        # 2. 训练配置
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.002)
        self.criterion = nn.MSELoss()
        self.dataset = []

        # 3. 仿真状态变量
        self.sim_mode = "IDLE"
        self.sim_data = {"pos": [], "vel": [], "acc": [], "target": []}

        # 4. 构建界面
        self.setup_ui()
        self.generate_default_curve()

        # 5. 定时器 (用于仿真循环，100ms步长)
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左侧控制面板 (30%) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(380)

        # A. 线路设置表格
        grp_line = QGroupBox("1. 线路限速配置 (ATP)")
        l_line = QVBoxLayout(grp_line)

        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["起点(m)", "终点(m)", "限速(km/h)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l_line.addWidget(self.table)

        btn_curve = QPushButton("生成/更新线路曲线")
        btn_curve.setStyleSheet("background-color: #e1e1e1; padding: 5px;")
        btn_curve.clicked.connect(self.generate_curve_from_table)
        l_line.addWidget(btn_curve)

        left_layout.addWidget(grp_line)

        # B. 控制中心
        grp_ctrl = QGroupBox("2. 仿真控制中心")
        l_ctrl = QVBoxLayout(grp_ctrl)

        self.btn_mpc = QPushButton("第一步：启动 MPC (生成专家数据)")
        self.btn_mpc.setStyleSheet("background-color: #ADD8E6; font-weight: bold; padding: 8px;")
        self.btn_mpc.clicked.connect(lambda: self.start_sim("MPC"))

        self.btn_train = QPushButton("第二步：训练 神经网络 (行为克隆)")
        self.btn_train.setStyleSheet("background-color: #98FB98; font-weight: bold; padding: 8px;")
        self.btn_train.clicked.connect(self.train_model)

        self.btn_ai = QPushButton("第三步：启动 AI (智能驾驶验证)")
        self.btn_ai.setStyleSheet("background-color: #FFB6C1; font-weight: bold; padding: 8px;")
        self.btn_ai.clicked.connect(lambda: self.start_sim("AI"))

        l_ctrl.addWidget(self.btn_mpc)
        l_ctrl.addWidget(self.btn_train)
        l_ctrl.addWidget(self.btn_ai)
        left_layout.addWidget(grp_ctrl)

        # C. 实时仪表盘
        grp_dash = QGroupBox("3. 实时状态监测")
        l_dash = QVBoxLayout(grp_dash)

        self.lbl_status = QLabel("系统就绪")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        self.lbl_pos = QLabel("位置: 0.0 m")
        self.lbl_speed = QLabel("速度: 0.0 km/h")
        self.lbl_acc = QLabel("加速度: 0.00 m/s²")
        self.lbl_acc.setStyleSheet("color: darkred; font-weight: bold;")  # 突出显示加速度

        l_dash.addWidget(self.lbl_status)
        l_dash.addWidget(self.lbl_pos)
        l_dash.addWidget(self.lbl_speed)
        l_dash.addWidget(self.lbl_acc)
        left_layout.addWidget(grp_dash)

        left_layout.addStretch()

        # --- 右侧绘图区 (70%) ---
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        r_layout.addWidget(self.canvas)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

    def generate_default_curve(self):
        """填入默认的高速线路数据"""
        defaults = [
            (0, 5000, 200),  # 起步提速区
            (5000, 10000, 350),  # 高速巡航区
            (10000, 12000, 160),  # 进站减速区
            (12000, 15000, 80)  # 站内低速区
        ]
        for r, (s, e, v) in enumerate(defaults):
            self.table.setItem(r, 0, QTableWidgetItem(str(s)))
            self.table.setItem(r, 1, QTableWidgetItem(str(e)))
            self.table.setItem(r, 2, QTableWidgetItem(str(v)))
        self.generate_curve_from_table()

    def generate_curve_from_table(self):
        """读取表格 -> 生成 TrackProfile -> 绘图"""
        self.track.reset_limits()
        try:
            for r in range(self.table.rowCount()):
                s_item = self.table.item(r, 0)
                e_item = self.table.item(r, 1)
                v_item = self.table.item(r, 2)
                if s_item and e_item and v_item:
                    self.track.apply_limit(float(s_item.text()), float(e_item.text()), float(v_item.text()))

            # 重新计算平滑制动曲线
            self.track.calculate_braking_curve()
            self.plot_curve()

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请确保表格中输入的是有效数字！")

    def plot_curve(self):
        """绘制静态参考曲线"""
        self.ax.clear()
        self.ax.plot(self.track.x, self.track.static_limit * 3.6, 'k--', alpha=0.3, label='ATP Limit (Safety)')
        self.ax.plot(self.track.x, self.track.target_curve * 3.6, 'r-', linewidth=1.5, label='Target Profile')
        self.ax.set_title("Speed Profile Planning")
        self.ax.set_ylabel("Speed (km/h)")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.canvas.draw()

    def start_sim(self, mode):
        """初始化并启动仿真"""
        self.sim_mode = mode
        self.dynamics = TrainDynamics()  # 重置物理模型状态
        self.curr_pos = 0.0
        self.curr_vel = 0.0
        self.sim_data = {"pos": [], "vel": [], "acc": [], "target": []}

        if mode == "MPC":
            self.dataset = []  # MPC模式清空旧数据重新采集
            self.lbl_status.setText("状态: MPC 专家控制运行中...")
        elif mode == "AI":
            if not self.dataset:
                QMessageBox.warning(self, "警告", "未检测到训练数据，AI 可能未经过训练！")
            self.lbl_status.setText("状态: AI 神经网络接管中...")

        self.timer.start(100)  # 100ms 刷新周期

    def game_loop(self):
        """核心仿真循环 (10Hz) - 带异常捕获版"""
        try:
            # 1. 结束条件判断
            if self.curr_pos >= self.total_distance:
                self.timer.stop()
                self.lbl_status.setText(f"状态: {self.sim_mode} 运行结束")
                if self.sim_mode == "MPC":
                    QMessageBox.information(self, "完成",
                                            f"已成功采集 {len(self.dataset)} 条专家驾驶数据。\n请点击'训练神经网络'。")
                return

            # 2. 获取当前目标
            target_v = self.track.get_target_v(self.curr_pos)

            # 3. 决策算法 (Controller)
            u = 0.0
            if self.sim_mode == "MPC":
                u = self.mpc.get_action(self.curr_pos, self.curr_vel)
                # 数据收集
                state = [self.curr_vel / 100.0, (target_v - self.curr_vel) / 20.0]
                self.dataset.append((state, u))

            elif self.sim_mode == "AI":
                state = torch.tensor([[self.curr_vel / 100.0, (target_v - self.curr_vel) / 20.0]], dtype=torch.float32)
                with torch.no_grad():
                    u = self.net(state).item()

            # 4. 物理环境更新
            # 注意：请确保 dynamics.step 返回 3 个值。如果你的 step 只返回 2 个值，这里会报错！
            self.curr_pos, self.curr_vel, _ = self.dynamics.step(self.curr_pos, self.curr_vel, u)

            # 5. 计算加速度 (高精度版)
            current_force = self.dynamics.current_force
            res = 1.0 + 0.01 * self.curr_vel + 0.0008 * (self.curr_vel ** 2)

            # 避免起步死区显示波动
            if self.curr_vel < 0.1 and 0 < current_force <= 3.0:
                real_acc = 0.0
            else:
                real_acc = (current_force - res) / self.dynamics.mass

            # 6. 数据记录
            self.sim_data["pos"].append(self.curr_pos)
            self.sim_data["vel"].append(self.curr_vel * 3.6)
            self.sim_data["target"].append(target_v * 3.6)

            # 7. 界面刷新
            if len(self.sim_data["pos"]) % 2 == 0:
                self.lbl_pos.setText(f"位置: {self.curr_pos:.1f} m")
                self.lbl_speed.setText(f"速度: {self.curr_vel * 3.6:.1f} km/h")
                self.lbl_acc.setText(f"加速度: {real_acc:.2f} m/s²")

            if len(self.sim_data["pos"]) % 5 == 0:
                self.update_plot()

        except Exception as e:
            # === 如果出错，程序不会闪退，而是会打印错误信息并停止计时器 ===
            self.timer.stop()
            import traceback
            error_msg = traceback.format_exc()
            print("【严重错误捕获】:")
            print(error_msg)
            QMessageBox.critical(self, "程序发生内部错误", f"错误详情:\n{e}\n\n请查看控制台输出以修复 Bug。")
    def update_plot(self):
        """动态曲线绘制"""
        self.ax.clear()
        # 绘制背景目标线
        self.ax.plot(self.track.x, self.track.target_curve * 3.6, 'r-', alpha=0.4, label='Target')
        # 绘制实际运行线
        self.ax.plot(self.sim_data["pos"], self.sim_data["vel"], 'b-', linewidth=2, label='Actual Speed')

        self.ax.set_ylim(0, 400)  # 适应高速列车量程
        self.ax.set_ylabel("Speed (km/h)")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.canvas.draw()

    def train_model(self):
        """神经网络训练逻辑"""
        if not self.dataset:
            QMessageBox.warning(self, "无数据", "请先运行 MPC 模式以采集数据！")
            return

        self.lbl_status.setText("状态: 神经网络训练中 (Epoch 0/500)...")
        QApplication.processEvents()  # 强制刷新界面避免假死

        # 数据预处理
        X = torch.tensor([d[0] for d in self.dataset], dtype=torch.float32)
        Y = torch.tensor([[d[1]] for d in self.dataset], dtype=torch.float32)

        self.net.train()
        epochs = 500
        for e in range(epochs):
            self.optimizer.zero_grad()
            output = self.net(X)
            loss = self.criterion(output, Y)
            loss.backward()
            self.optimizer.step()

            if e % 50 == 0:
                self.lbl_status.setText(f"状态: 神经网络训练中 (Epoch {e}/{epochs})... Loss={loss.item():.4f}")
                QApplication.processEvents()

        self.lbl_status.setText(f"状态: 训练完成! 最终 Loss: {loss.item():.5f}")
        QMessageBox.information(self, "成功",
                                f"模型训练完成！\n最终 Loss: {loss.item():.5f}\n现在可以点击'启动 AI'进行测试。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())