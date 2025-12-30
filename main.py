import sys
import os

# === 关键修复 1：强制指定 Matplotlib 后端，防止崩溃 ===
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication
from gui_app import MainWindow

if __name__ == "__main__":
    # === 关键修复 2：解决 Anaconda 的 Qt 插件路径问题 ===
    try:
        import PyQt5
        plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    except:
        pass

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())