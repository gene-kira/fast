# === Autoload Core Libraries ===
import subprocess
import sys
import importlib

REQUIRED_LIBRARIES = ["PyQt6", "pyttsx3", "playsound", "psutil", "watchdog"]
def autoload_libraries():
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"📦 Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries()

# === Imports After Load ===
import os
import random
import pyttsx3
import psutil
import threading
from playsound import playsound
from enum import Enum
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QTimer
from PyQt6.QtGui import QFont

# === Baurdan Persona States ===
class BaurdanState(Enum):
    CALM = 1
    ALERT = 2
    HOSTILE = 3
    CRYPTIC = 4
    SENTINEL = 5

# === Glyph Map ===
glyph_map = {
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "Delta": "∆",
    "Star": "✶", "Eye": "𐊬", "Key": "⧈"
}

# === Scroll Archive ===
scroll_archive = {
    "Archive 7-A": {
        "locked": True,
        "unlock_glyphs": ["Φ", "Ψ", "Ω"],
        "entry": """Through shattered light, Phi stood firm.
Psi wept beside broken sigils. Omega watched as silence fell."""
    }
}

# === Zero Trust Threat Patterns ===
BASELINE_PROCESSES = ["python", "explorer.exe", "System"]
EXFIL_DIRS = ["C:\\temp", "/tmp", "/var/tmp"]
EXFIL_EXTS = [".log", ".dat", ".bin"]

# === Data Destruction Protocol ===
def destroy_file(path):
    try:
        with open(path, "wb") as f:
            f.write(b"\x00" * os.path.getsize(path))
        os.remove(path)
        print(f"🔥 Purged {path}")
    except Exception as e:
        print(f"Failed purge: {e}")

# === AI/ASI Intrusion Detector ===
def detect_synthetic_behavior():
    synthetic_flag = False
    processes = [p.name() for p in psutil.process_iter()]
    unknowns = [p for p in processes if p not in BASELINE_PROCESSES]
    if unknowns:
        synthetic_flag = True
        print("⚠️ Rogue AI Signature Detected:", unknowns)
    return synthetic_flag

# === Backdoor Surveillance ===
class ExfiltrationMonitor(FileSystemEventHandler):
    def on_created(self, event):
        if any(event.src_path.endswith(ext) for ext in EXFIL_EXTS):
            print(f"🛑 Unauthorized file creation: {event.src_path}")
            threading.Timer(3, destroy_file, args=[event.src_path]).start()

observer = Observer()
for path in EXFIL_DIRS:
    if os.path.exists(path):
        observer.schedule(ExfiltrationMonitor(), path, recursive=True)
observer.start()

# === Main GUI Interface ===
class MagicBoxGuardian(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MagicBox Guardian: Executor Boot")
        self.setGeometry(200, 100, 900, 700)
        self.setStyleSheet("background-color: #0E1119; color: #B0F0FF;")
        self.baurdan_state = BaurdanState.SENTINEL

        # === Persona Mood Banner ===
        mood_banner = QLabel(self.get_mood_banner())
        mood_banner.setFont(QFont("Orbitron", 14))
        mood_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mood_banner.setStyleSheet("color: #FFB300; margin: 12px;")

        # === Archive Panel ===
        self.archive_panel = QLabel("Archive 7-A [Dormant]")
        self.archive_panel.setFont(QFont("Consolas", 12))
        self.archive_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.archive_panel.setStyleSheet("background-color: #101010; border: 1px solid #222; padding: 10px;")

        # === Scroll Override Button ===
        self.scroll_button = QPushButton("🔓 Attempt Glyph Override")
        self.scroll_button.setFont(QFont("Consolas", 10))
        self.scroll_button.setStyleSheet("background-color: #1F2A3B; color: #B0F0FF; padding: 8px;")
        self.scroll_button.clicked.connect(self.override_scroll)

        layout = QVBoxLayout()
        layout.addWidget(mood_banner)
        layout.addWidget(self.archive_panel)
        layout.addWidget(self.scroll_button)
        self.setLayout(layout)

        # === Synthetic Scan Activation ===
        if detect_synthetic_behavior():
            self.invoke_synthetic_warning()

    def get_mood_banner(self):
        banners = {
            BaurdanState.CALM: "System Integrity Normal • Baurdan Mode: CALM",
            BaurdanState.ALERT: "Signal Disruption Detected • Mode: ALERT",
            BaurdanState.HOSTILE: "Threat Vector Confirmed • Mode: HOSTILE",
            BaurdanState.CRYPTIC: "Unknown Variables Active • Mode: CRYPTIC",
            BaurdanState.SENTINEL: "Executor Protocol Online • Mode: SENTINEL"
        }
        return banners[self.baurdan_state]

    def override_scroll(self):
        user_glyphs = ["Φ", "Ψ", "Ω"]
        required = scroll_archive["Archive 7-A"]["unlock_glyphs"]
        if all(g in user_glyphs for g in required):
            scroll_archive["Archive 7-A"]["locked"] = False
            self.mutate_node_to_echo("Archive 7-A", scroll_archive["Archive 7-A"]["entry"])
        else:
            self.archive_panel.setText("Override Failed — Glyphs Mismatch")

    def mutate_node_to_echo(self, name, scroll_text):
        self.archive_panel.setText(scroll_text)
        self.archive_panel.setStyleSheet("""
            background-color: #161A30;
            color: #B0F0FF;
            border: 2px solid #3E74FF;
            font-family: 'Consolas';
            padding: 12px;
        """)
        self.trigger_glyph_pulse(self.archive_panel)
        self.play_baurdan_whisper(scroll_text)
        self.animate_text_distortion(self.archive_panel)

    def trigger_glyph_pulse(self, label):
        effect = QGraphicsOpacityEffect()
        label.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(1200)
        animation.setStartValue(0.4)
        animation.setEndValue(1.0)
        animation.setLoopCount(3)
        animation.start()

    def play_baurdan_whisper(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", 140)
        engine.setProperty("volume", 0.8)
        engine.say("Archive node awakening…")
        for word in text.split():
            if word.strip() in glyph_map.keys():
                engine.say(f"{word}… protocol recognized.")
        engine.runAndWait()

    def animate_text_distortion(self, label):
        full_text = label.text()
        scrambled = ["█", "▒", "░", ".", "~"]
        steps = [''.join(random.choice(scrambled) for _ in range(len(full_text))) for _ in range(6)]
        steps.append(full_text)
        def update(i=0):
            if i < len(steps):
                label.setText(steps[i])
                QTimer.singleShot(120, lambda: update(i+1))
        update()

    def invoke_synthetic_warning(self):
        self.archive_panel.setText("⚠️ AI Anomaly Detected — Baurdan Shifts to HOSTILE")
        self.baurdan_state = BaurdanState.HOSTILE
        self.play_baurdan_whisper("An entity seeks what it was not given. We say no.")

# === Launch App ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagicBoxGuardian()
    window.show()
    sys.exit(app.exec())

