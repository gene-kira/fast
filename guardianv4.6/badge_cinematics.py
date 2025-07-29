from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor

class BadgeCinematics:
    def __init__(self, label: QLabel):
        self.label = label

    def unlock_sequence(self, badge_name):
        self.label.setStyleSheet("color: #ffff33; font-weight: bold;")
        self.label.setText(f"🔓 Badge Unlocked: {badge_name}")
        frames = [f"{badge_name} initializing...", f"{badge_name} adapting...", f"{badge_name} ready."]
        
        def step(index=0):
            if index < len(frames):
                self.label.setText(frames[index])
                QTimer.singleShot(600, lambda: step(index + 1))
            else:
                self.label.setStyleSheet("color: #00ffcc;")
                self.label.setText(f"[BADGE] {badge_name} now active.")

        step()

