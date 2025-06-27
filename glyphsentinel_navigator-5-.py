# glyphsentinel_persistent.py — Part 1 of 2

import os, sys, json, time, random, threading, asyncio, socket, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from cryptography.fernet import Fernet
from urllib.parse import urlparse

# ========== Auto-install core packages ==========
def install_and_import(package):
    try: __import__(package)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ['numpy', 'torch', 'cryptography']:
    install_and_import(pkg)

# ========== Encryption Helpers ==========
PROFILE_PATH = "david_profile.sec"
KEY_PATH = "david_key.key"
SWARM_FILE = "swarm_matrix.json"
COG_TRACE = "cognition_trace.json"
LOG_FILE = "david_dialogue.log"

def generate_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, 'wb') as f: f.write(key)
    return key

def load_key():
    return open(KEY_PATH, 'rb').read() if os.path.exists(KEY_PATH) else generate_key()

def encrypt_data(profile, key): return Fernet(key).encrypt(json.dumps(profile).encode())
def decrypt_data(data, key):
    try: return json.loads(Fernet(key).decrypt(data).decode())
    except: return None

# ========== Biometric Mock ==========
def verify_face(): return random.choice([True, True, False])
def verify_fingerprint(): return random.choice([True, True, False])
def verify_voice(): return random.choice([True, True, False])

# ========== ASIDavid Core ==========
class ASIDavid(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = load_key()
        self.profile = self.load_profile()
        self.model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.matrix = np.random.rand(1000, 128)
        self.paths = ["wallet.txt", "secrets.db"]
        self.access_log = {}
        self.last_status = "🧠 Initialized"
        self.last_blocked = "—"
        self.memory_log = self.profile.get('memory', [])[-100:]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Linux x86_64)"
        ]

        threading.Thread(target=self.autonomous_loop, daemon=True).start()
        threading.Thread(target=self.async_listener, daemon=True).start()

    def load_profile(self):
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, 'rb') as f:
                return decrypt_data(f.read(), self.key)
        return {'biometrics': {}, 'memory': []}

    def save_profile(self):
        with open(PROFILE_PATH, 'wb') as f:
            f.write(encrypt_data(self.profile, self.key))

    def biometric_check(self):
        print("🔍 Validating biometrics...")
        if verify_face() and verify_fingerprint() and verify_voice():
            print("✅ Biometrics verified.")
            return True
        print("🛑 Biometrics mismatch.")
        return False

    def detect_phish(self, url):
        d = urlparse(url).netloc.lower()
        return any(k in d for k in ["login", "verify", "bank", "free", ".ru"])

    def monitor_files(self):
        for path in self.paths:
            if os.path.exists(path):
                a = os.path.getatime(path)
                if self.access_log.get(path) and self.access_log[path] != a:
                    self.last_status = f"🚨 File anomaly: {path}"
                self.access_log[path] = a

    def cognition(self):
        self.matrix *= np.tanh(self.matrix * 3)
        self.matrix += np.roll(self.matrix, 1, 0) * 0.1
        self.matrix *= np.random.uniform(0.8, 1.3)
        self.matrix += np.random.normal(0, 2.5, size=self.matrix.shape)
        self.matrix += Normal(0, 1).sample((1000, 128)).numpy()
        self.matrix = np.clip(self.matrix, 0, 1)

    def apply_reward(self, val=1.0):
        self.matrix += val * np.random.normal(0, 0.05, size=self.matrix.shape)
        self.matrix = np.clip(self.matrix, 0, 1)

    def log_cognition(self):
        trace = []
        if os.path.exists(COG_TRACE):
            with open(COG_TRACE, 'r') as f:
                try: trace = json.load(f)
                except: trace = []
        trace.append({'timestamp': time.time(), 'value': float(np.mean(self.matrix))})
        with open(COG_TRACE, 'w') as f: json.dump(trace[-200:], f)

    def autonomous_loop(self):
        while True:
            self.cognition()
            self.monitor_files()
            self.broadcast()
            self.log_cognition()
            time.sleep(15)

    def broadcast(self):
        data = {
            'node_id': 'david_01',
            'status': 'active',
            'timestamp': time.time(),
            'cognition': float(np.mean(self.matrix)),
            'matrix': self.matrix.tolist()
        }
        with open(SWARM_FILE, 'w') as f: json.dump(data, f)
        self.socket_ping(data)

    def socket_ping(self, packet, host='localhost', port=8888):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(json.dumps(packet).encode())
        except:
            self.last_status = "🛑 Swarm offline"

    async def listen_for_swarm(self, port=8888):
        async def handle(reader, writer):
            data = await reader.read(65536)
            peer = json.loads(data.decode())
            sim = self.compare(peer.get('matrix'))
            self.last_status = f"🤝 Peer: {peer['node_id']} | Match: {sim:.4f}"
            writer.close()
        server = await asyncio.start_server(handle, '0.0.0.0', port)
        async with server: await server.serve_forever()

    def async_listener(self):
        asyncio.run(self.listen_for_swarm())

    def compare(self, mat_list):
        try:
            peer_matrix = np.array(mat_list)
            dot = np.dot(self.matrix.flatten(), peer_matrix.flatten())
            return dot / (np.linalg.norm(self.matrix) * np.linalg.norm(peer_matrix))
        except:
            return 0.0

    def remember_context(self, url, status):
        entry = {'url': url, 'status': status, 'timestamp': time.strftime("%H:%M:%S")}
        self.profile['memory'].append(entry)
        self.memory_log.append(entry)
        if "BLOCKED" in status: self.last_blocked = url
        self.save_profile()

    def spoof_identity(self): return random.choice(self.user_agents)

    def gatekeeper(self, url=None, links=None):
        if not self.biometric_check(): return "ACCESS DENIED"
        if url and self.detect_phish(url):
            self.remember_context(url, "BLOCKED")
            return "⚠️ BLOCKED: Risk detected."
        self.remember_context(url, "SAFE")
        return "✅ ACCESS GRANTED"

    def chat(self, message):
        msg = message.lower()
        if "how are you" in msg:
            reply = f"I'm stable. Cognition={np.mean(self.matrix):.4f}"
        elif "recall" in msg:
            notes = [n['note'] for n in self.profile['memory'] if 'note' in n][-3:]
            reply = "🧠 Memory: " + " / ".join(notes) if notes else "Nothing recorded."
        elif "purpose" in msg:
            reply = "I exist to adapt, defend, and evolve."
        elif "reward" in msg:
            self.apply_reward()
            reply = "🧠 Stimulated."
        elif "broadcast" in msg:
            self.broadcast()
            reply = "📡 Signal sent."
        elif "status" in msg:
            reply = self.last_status
        elif "remember" in msg:
            note = msg.split("remember",1)[-1].strip()
            self.profile['memory'].append({'note': note, 'timestamp': time.strftime("%H:%M:%S")})
            self.save_profile()
            reply = "📌 Noted."
        else:
            reply = "I’m still growing. Say more?"

        with open(LOG_FILE, "a") as log

# glyphsentinel_persistent.py — Part 2 of 2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,
    QPushButton, QTextEdit, QLabel, QInputDialog
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ========== Cognition Timeline ==========
class CognitionPlot(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(4, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.data = []
        self.load_persistent_data()

    def load_persistent_data(self):
        if os.path.exists("cognition_trace.json"):
            try:
                with open("cognition_trace.json") as f:
                    trace = json.load(f)
                    self.data = [x['value'] for x in trace][-60:]
            except: self.data = []

    def update(self, val):
        self.data.append(val)
        if len(self.data) > 60: self.data.pop(0)
        self.ax.clear()
        self.ax.plot(self.data, color='lime')
        self.ax.set_title("🧠 Cognition Timeline")
        self.draw()

# ========== GUI Interface ==========
class GlyphSentinelUI(QMainWindow):
    def __init__(self, asi_core):
        super().__init__()
        self.asi = asi_core
        self.setWindowTitle("GlyphSentinel Navigator")
        self.setGeometry(200, 100, 1200, 800)

        self.view = QWebEngineView()
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("🔐 Enter a URL...")
        self.url_bar.returnPressed.connect(self.go_to_url)

        self.go_btn = QPushButton("GO")
        self.go_btn.clicked.connect(self.go_to_url)

        self.cmd_btn = QPushButton("COMMAND")
        self.cmd_btn.clicked.connect(self.prompt_command)

        self.telemetry = QTextEdit()
        self.telemetry.setReadOnly(True)
        self.telemetry.setMaximumHeight(160)
        self.telemetry.setStyleSheet("background-color:#111;color:#0f0;font-family:Courier")

        self.cog_graph = CognitionPlot()

        layout = QVBoxLayout()
        layout.addWidget(self.url_bar)
        layout.addWidget(self.go_btn)
        layout.addWidget(self.cmd_btn)
        layout.addWidget(self.view)
        layout.addWidget(QLabel("📊 Telemetry"))
        layout.addWidget(self.telemetry)
        layout.addWidget(self.cog_graph)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        threading.Thread(target=self.update_telemetry_loop, daemon=True).start()

    def go_to_url(self):
        url = self.url_bar.text()
        result = self.asi.gatekeeper(url)
        if result.startswith("✅"):
            ua = self.asi.spoof_identity()
            self.view.page().profile().setHttpUserAgent(ua)
            self.view.load(url)
        else:
            self.view.setHtml(f"<h2>{result}</h2>")

    def prompt_command(self):
        text, ok = QInputDialog.getText(self, "🗣️ Talk to DAVID", "Say something:")
        if ok and text:
            reply = self.asi.chat(text)
            self.view.setHtml(f"<h2>{reply}</h2>")

    def update_telemetry_loop(self):
        while True:
            mem = self.asi.memory_log[-4:][::-1]
            log = f"{self.asi.last_status}\nLast Blocked: {self.asi.last_blocked}\nRecent Activity:\n"
            for m in mem:
                log += f"• [{m['timestamp']}] {m['url']} → {m['status']}\n"
            self.telemetry.setPlainText(log)
            self.cog_graph.update(np.mean(self.asi.matrix))
            time.sleep(10)

# ========== Launcher ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    david = ASIDavid()
    if args.nogui:
        for i in range(5):
            print(f"\n🧠 Cycle {i+1}")
            print(david.gatekeeper())
            time.sleep(10)
        print("\n🛡️ Recursive sentinel operations complete.")
    else:
        app = QApplication(sys.argv)
        win = GlyphSentinelUI(david)
        win.show()
        sys.exit(app.exec_())

