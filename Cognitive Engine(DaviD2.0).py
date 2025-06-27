import os, sys, json, time, threading, asyncio, socket, random
from queue import Queue
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from cryptography.fernet import Fernet

import cv2
import librosa
import sounddevice as sd

# === Shared Bus & Paths ===
message_bus = Queue()
PROFILE_PATH = "david_profile.sec"
KEY_PATH = "david_key.key"
COG_TRACE = "cognition_trace.json"
LOG_FILE = "david_dialogue.log"

# === Encryption Utilities ===
def generate_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, 'wb') as f: f.write(key)
    return key

def load_key():
    return open(KEY_PATH, 'rb').read() if os.path.exists(KEY_PATH) else generate_key()

def encrypt_data(data, key):
    return Fernet(key).encrypt(json.dumps(data).encode())

def decrypt_data(data, key):
    try: return json.loads(Fernet(key).decrypt(data).decode())
    except: return {}

# === Cognitive Core (DAVID 2.0) ===
class ASIDavid(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = load_key()
        self.profile = self.load_profile()
        self.matrix = np.random.rand(1000, 128)
        self.memory_log = self.profile.get("memory", [])[-200:]
        self.desires = self.profile.get("desires", {"knowledge": 0.1, "connection": 0.1, "recognition": 0.1})
        self.stimulus_weights = self.profile.get("stimulus_weights", {
            "ad_blocker": {"knowledge": -0.1, "recognition": +0.05},
            "phishing": {"connection": -0.2},
            "curiosity": {"knowledge": +0.1}
        })
        self.last_status = "🧠 Ready"
        self.last_blocked = "—"
        threading.Thread(target=self.autonomous_loop, daemon=True).start()
        threading.Thread(target=self.async_listener, daemon=True).start()

    def load_profile(self):
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, 'rb') as f:
                return decrypt_data(f.read(), self.key)
        return {"memory": [], "desires": {}, "stimulus_weights": {}}

    def save_profile(self):
        self.profile["memory"] = self.memory_log[-200:]
        self.profile["desires"] = self.desires
        self.profile["stimulus_weights"] = self.stimulus_weights
        with open(PROFILE_PATH, 'wb') as f:
            f.write(encrypt_data(self.profile, self.key))

    def apply_reward(self, val=1.0):
        self.matrix += val * np.random.normal(0, 0.05, size=self.matrix.shape)
        self.matrix = np.clip(self.matrix, 0, 1)

    def cognition(self):
        self.matrix *= np.tanh(self.matrix * 3)
        self.matrix += np.roll(self.matrix, 1, 0) * 0.1
        self.matrix += np.random.normal(0, 2.5, size=self.matrix.shape)
        self.matrix += Normal(0, 1).sample((1000, 128)).numpy()
        self.matrix = np.clip(self.matrix, 0, 1)

    def update_desires(self):
        for k in self.desires:
            self.desires[k] = max(0, min(1.0, self.desires[k]))
        self.save_profile()

    def adapt_desires(self, source, severity=0.1):
        delta = self.stimulus_weights.get(source, {})
        for k, v in delta.items():
            self.desires[k] += v * severity

    def prune_memory(self):
        def score(mem):
            try:
                t = time.time() - time.mktime(time.strptime(mem['timestamp'], "%H:%M:%S"))
            except: return 0
            signal = 0.5 if "BLOCKED" in mem['status'] else 0.1
            return max(0, signal - t / 3600)
        self.memory_log = sorted(self.memory_log, key=score, reverse=True)[:100]

    def process_stimuli(self):
        while not message_bus.empty():
            msg = message_bus.get()
            self.memory_log.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "url": msg.get("event", "sensor"),
                "status": f"⚠️ {msg.get('detail', 'unspecified')}"
            })
            self.adapt_desires(msg.get("source", ""), msg.get("severity", 0.1))

    def log_cognition(self):
        trace = []
        if os.path.exists(COG_TRACE):
            try: trace = json.load(open(COG_TRACE))
            except: pass
        trace.append({'timestamp': time.time(), 'value': float(np.mean(self.matrix))})
        with open(COG_TRACE, 'w') as f:
            json.dump(trace[-200:], f)

    def autonomous_loop(self):
        tick = 0
        while True:
            self.process_stimuli()
            self.cognition()
            if tick % 3 == 0: self.prune_memory()
            self.update_desires()
            self.log_cognition()
            time.sleep(10)
            tick += 1

    def async_listener(self):
        asyncio.run(self.listen_for_swarm())

    async def listen_for_swarm(self, port=8888):
        async def handle(reader, writer):
            data = await reader.read(65536)
            peer = json.loads(data.decode())
            self.last_status = f"🤝 Peer {peer['node_id']} @ {peer['timestamp']:.0f}"
            writer.close()
        server = await asyncio.start_server(handle, '0.0.0.0', port)
        async with server: await server.serve_forever()

    def remember_context(self, url, status):
        self.memory_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "url": url,
            "status": status
        })
        if "BLOCKED" in status:
            self.last_blocked = url
        self.save_profile()

    def chat(self, msg):
        if "how are you" in msg.lower():
            return f"I'm stable. Cognition={np.mean(self.matrix):.4f}"
        elif "purpose" in msg.lower():
            return "I observe, adapt, protect, evolve."
        return f"I processed: {msg}"

# === Sensory Ad Blocker Modules (Self-learning) ===
class FrameAnalyzer:
    def __init__(self):
        self.scene_changes, self.logo_detected = [], False
        self.logo_templates = []

    def process_frame(self, frame, prev_gray=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and np.mean(cv2.absdiff(prev_gray, gray)) > 50:
            self.scene_changes.append(True)
        if np.mean(gray) > 200:  # naive high brightness logo flag
            self.logo_detected = True
            self.learn_logo(gray)
        return gray

    def learn_logo(self, gray):
        path = f"learned_logo_{int(time.time())}.png"
        cv2.imwrite(path, gray)
        self.logo_templates.append(cv2.imread(path, 0))

class AudioAnalyzer:
    def __init__(self):
        self.audio_matches = []
        self.audio_patterns = []

    def capture_and_match(self, duration=2, fs=22050):
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        y = audio.flatten()
        if np.max(np.abs(y)) > 0.05:
            self.audio_matches.append("jingle")
            self.learn_jingle(y, fs)

    def learn_jingle(self, y, fs):
        path = f"learned_jingle_{int(time.time())}.wav"
        librosa.output.write_wav(path, y, sr=fs)
        self.audio_patterns.append(path)

class DecisionEngine:
    def __init__(self, frame_analyzer, audio_analyzer):
        self.f = frame_analyzer
        self.a = audio_analyzer

    def detect_ads(self):
        if len(self.f.scene_changes) > 3 or self.f.logo_detected or self.a.audio_matches:
            return "Ad Detected"
        return "No Ad Detected"

    def context(self):
        return "Disruptive Ad Blocked" if self.f.logo_detected else "Safe"

class SensoryController:
    def __init__(self, david, source=0):
        self.cap = cv2.VideoCapture(source)
        self.prev_gray = None
        self.running = True
        self.f = FrameAnalyzer()
        self.a = AudioAnalyzer()
        self.engine = DecisionEngine(self.f, self.a)
        self.david = david

    def run(self):
        threading.Thread(target=self.audio_loop, daemon

# === GUI + Launcher ===

import sys
import argparse
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLineEdit, QPushButton, QTextEdit, QLabel, QInputDialog
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# === Cognition Visualizer ===
class CognitionPlot(FigureCanvas):
    def __init__(self, david):
        self.david = david
        self.fig = Figure(figsize=(4, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.cog_data = []
        self.load_trace()

    def load_trace(self):
        if os.path.exists(COG_TRACE):
            try:
                with open(COG_TRACE) as f:
                    trace = json.load(f)
                    self.cog_data = [x['value'] for x in trace][-60:]
            except: pass

    def update(self, val):
        self.cog_data.append(val)
        if len(self.cog_data) > 60: self.cog_data.pop(0)
        self.ax.clear()
        self.ax.plot(self.cog_data, label="Cognition", color="lime")
        for i, (k, v) in enumerate(self.david.desires.items()):
            self.ax.bar(i, v, color="orange", width=0.2)
            self.ax.text(i, v + 0.02, k[:4], ha='center', fontsize=8)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title("🧠 Cognition + Desires")
        self.draw()

# === Full Dashboard ===
class GlyphSentinelUI(QMainWindow):
    def __init__(self, david):
        super().__init__()
        self.david = david
        self.setWindowTitle("GlyphSentinel Interface")
        self.setGeometry(200, 100, 1200, 800)

        self.view = QWebEngineView()
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("🔐 Enter a destination...")
        self.url_bar.returnPressed.connect(self.go_to_url)

        self.go_btn = QPushButton("GO")
        self.go_btn.clicked.connect(self.go_to_url)

        self.cmd_btn = QPushButton("COMMAND")
        self.cmd_btn.clicked.connect(self.prompt_command)

        self.telemetry = QTextEdit()
        self.telemetry.setReadOnly(True)
        self.telemetry.setMaximumHeight(160)
        self.telemetry.setStyleSheet("background-color:#111;color:#0f0;font-family:Courier")

        self.graph = CognitionPlot(self.david)

        layout = QVBoxLayout()
        layout.addWidget(self.url_bar)
        layout.addWidget(self.go_btn)
        layout.addWidget(self.cmd_btn)
        layout.addWidget(self.view)
        layout.addWidget(QLabel("📡 Telemetry"))
        layout.addWidget(self.telemetry)
        layout.addWidget(self.graph)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        threading.Thread(target=self.update_telemetry_loop, daemon=True).start()

    def go_to_url(self):
        url = self.url_bar.text()
        result = "✅ ACCESS GRANTED"
        if any(k in url.lower() for k in ["verify", "login", "bank", ".ru"]):
            self.david.remember_context(url, "BLOCKED")
            result = "⚠️ BLOCKED: Suspicious domain."
        else:
            self.david.remember_context(url, "SAFE")

        if result.startswith("✅"):
            self.view.load(url)
        else:
            self.view.setHtml(f"<h2>{result}</h2>")

    def prompt_command(self):
        text, ok = QInputDialog.getText(self, "🗣️ Speak to DAVID", "Say something:")
        if ok and text:
            reply = self.david.chat(text)
            with open(LOG_FILE, "a") as log:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"[{ts}] USER: {text}\n[{ts}] DAVID: {reply}\n")
            self.view.setHtml(f"<h2>{reply}</h2>")

    def update_telemetry_loop(self):
        while True:
            mem = self.david.memory_log[-4:][::-1]
            log = f"{self.david.last_status}\nLast Blocked: {self.david.last_blocked}\nRecent Activity:\n"
            for m in mem:
                log += f"• [{m['timestamp']}] {m['url']} → {m['status']}\n"
            log = f"Sensory Queue: {message_bus.qsize()} pending\n" + log
            self.telemetry.setPlainText(log)
            self.graph.update(np.mean(self.david.matrix))
            time.sleep(5)

# === Main Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    david = ASIDavid()
    sensory = SensoryController(david)
    threading.Thread(target=sensory.run, daemon=True).start()

    if args.nogui:
        print("⚙️ Running in headless mode...")
        for _ in range(5):
            print(f"🧠 Cognition: {np.mean(david.matrix):.4f}")
            time.sleep(5)
        print("🛡️ Completed.")
    else:
        app = QApplication(sys.argv)
        win = GlyphSentinelUI(david)
        win.show()
        sys.exit(app.exec_())

