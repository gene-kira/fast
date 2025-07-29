import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from guardian_engine import MagicBoxGuardian

class GuardianGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.guardian = MagicBoxGuardian()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MAGICBOX GUARDIAN v4.5")
        self.setGeometry(200, 200, 700, 500)
        self.setStyleSheet("background-color: #121212; color: #00ffe7;")
        layout = QVBoxLayout()
        
        self.status_panel = QLabel("STATUS: Baurdan Active")
        self.status_panel.setFont(QFont("Arial", 14))
        layout.addWidget(self.status_panel)

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Enter suspicious IP")
        layout.addWidget(self.ip_input)

        detect_btn = QPushButton("Detect Threat")
        detect_btn.clicked.connect(self.detect_threat)
        layout.addWidget(detect_btn)

        self.badge_input = QLineEdit()
        self.badge_input.setPlaceholderText("Enter badge name")
        layout.addWidget(self.badge_input)

        badge_btn = QPushButton("Unlock Badge")
        badge_btn.clicked.connect(self.unlock_badge)
        layout.addWidget(badge_btn)

        self.override_input = QLineEdit()
        self.override_input.setPlaceholderText("Enter override key")
        layout.addWidget(self.override_input)

        override_btn = QPushButton("Override Persona")
        override_btn.clicked.connect(self.override_persona)
        layout.addWidget(override_btn)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        self.setLayout(layout)
        self.cinematic_boot()

    def cinematic_boot(self):
        self.log_view.setText("🔮 Initializing MAGICBOX GUARDIAN...\nPersona loading: Baurdan\nLoading shield protocol...")
        QTimer.singleShot(2000, lambda: self.log_view.append("🛡️ Guardian online. Threat matrix engaged."))

    def detect_threat(self):
        ip = self.ip_input.text()
        self.guardian.detect_threat(ip)
        self.refresh_logs()

    def unlock_badge(self):
        badge = self.badge_input.text()
        self.guardian.unlock_badge(badge)
        self.refresh_logs()

    def override_persona(self):
        key_attempt = self.override_input.text()
        self.guardian.persona_override("Baurdan", key_attempt)
        self.status_panel.setText(f"STATUS: {self.guardian.active_persona} Active")
        self.refresh_logs()

    def refresh_logs(self):
        self.log_view.setText("\n".join(self.guardian.logs))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GuardianGUI()
    gui.show()
    sys.exit(app.exec_())

