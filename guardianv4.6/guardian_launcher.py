from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class GuardianLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardian Boot Console")
        self.setGeometry(300, 300, 400, 250)
        self.setStyleSheet("background-color: #0f0f0f; color: #00ffcc;")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.title = QLabel("🔮 Guardian is Rising...")
        self.title.setFont(QFont("Orbitron", 14))
        layout.addWidget(self.title)

        self.status = QLabel("Checking Environment...")
        layout.addWidget(self.status)

        self.launch_btn = QPushButton("Ignite Core")
        self.launch_btn.clicked.connect(self.launch_gui)
        layout.addWidget(self.launch_btn)

        self.setLayout(layout)
        QTimer.singleShot(1500, lambda: self.status.setText("✅ All Systems Nominal"))

    def launch_gui(self):
        from guardian_gui import GuardianGUI
        self.hide()
        gui = GuardianGUI()
        gui.show()

if __name__ == "__main__":
    app = QApplication([])
    launcher = GuardianLauncher()
    launcher.show()
    app.exec_()

