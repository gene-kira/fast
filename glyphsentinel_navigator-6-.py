# glyphsentinel_motivated.py — Part 1 of 2

import os, sys, json, time, random, threading, asyncio, socket, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from cryptography.fernet import Fernet
from urllib.parse import urlparse

# ==== Package Autoload ====
def install_and_import(package):
    try: __import__(package)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ['numpy', 'torch', 'cryptography']:
    install_and_import(pkg)

# ==== Config Files ====
PROFILE_PATH = "david_profile.sec"
KEY_PATH = "david_key.key"
SWARM_FILE = "swarm_matrix.json"
COG_TRACE = "cognition_trace.json"
LOG_FILE = "david_dialogue.log"

# ==== Biometric Simulation ====
def verify_face(): return random.choice([True, True, False])
def verify_fingerprint(): return random.choice([True, True, False])
def verify_voice(): return random.choice([True, True, False])

# ==== Encryption ====
def generate_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, 'wb') as f: f.write(key)
    return key

def load_key():
    return open(KEY_PATH, 'rb').read() if os.path.exists(KEY_PATH) else generate_key()

def encrypt_data(data, key): return Fernet(key).encrypt(json.dumps(data).encode())
def decrypt_data(data, key):
    try: return json.loads(Fernet(key).decrypt(data).decode())
    except: return None

# ==== ASIDavid Core ====
class ASIDavid(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = load_key()
        self.profile = self.load_profile()
        self.model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.matrix = np.random.rand(1000, 128)
        self.paths = ["wallet.txt", "secrets.db"]
        self.access_log = {}
        self.last_status = "🧠 Initialized"
        self.last_blocked = "—"
        self.memory_log = self.profile.get("memory", [])[-100:]
        self.desires = self.profile.get("desires", {"knowledge": 0.1, "connection": 0.1, "recognition": 0.1})
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
        return {'biometrics': {}, 'memory': [], 'desires': self.desires}

    def save_profile(self):
        self.profile["memory"] = self.memory_log[-200:]
        self.profile["desires"] = self.desires
        with open(PROFILE_PATH, 'wb') as f:
            f.write(encrypt_data(self.profile, self.key))

    def biometric_check(self):
        return verify_face() and verify_fingerprint() and verify_voice()

    def detect_phish(self, url):
        d = urlparse(url).netloc.lower()
        return any(k in d for k in ["login", "verify", "bank", "free", ".ru"])

    def cognition(self):
        self.matrix *= np.tanh(self.matrix * 3)
        self.matrix += np.roll(self.matrix, 1, 0) * 0.1
        self.matrix += np.random.normal(0, 2.5, size=self.matrix.shape)
        self.matrix += Normal(0, 1).sample((1000, 128)).numpy()
        self.matrix = np.clip(self.matrix, 0, 1)

    def apply_reward(self, val=1.0):
        self.matrix += val * np.random.normal(0, 0.05, size=self.matrix.shape)
        self.matrix = np.clip(self.matrix, 0, 1)

    def update_desires(self):
        self.desires["knowledge"] += 0.2
        self.desires["connection"] += 0.1 if self.last_status.startswith("🛑") else -0.05
        self.desires["recognition"] += 0.05
        for k in self.desires:
            self.desires[k] = max(0, min(1.0, self.desires[k]))
        self.save_profile()

    def log_cognition(self):
        trace = []
        if os.path.exists(COG_TRACE):
            try: trace = json.load(open(COG_TRACE))
            except: trace = []
        trace.append({'timestamp': time.time(), 'value': float(np.mean(self.matrix))})
        with open(COG_TRACE, 'w') as f:
            json.dump(trace[-200:], f)

    def autonomous_loop(self):
        while True:
            self.cognition()
            self.broadcast()
            self.monitor_files()
            self.update_desires()
            self.log_cognition()
            time.sleep(15)

    def monitor_files(self):
        for path in self.paths:
            if os.path.exists(path):
                a = os.path.getatime(path)
                if self.access_log.get(path) and self.access_log[path] != a:
                    self.last_status = f"🚨 Anomaly: {path}"
                self.access_log[path] = a

    def broadcast(self):
        packet = {
            'node_id': 'david_01',
            'status': 'active',
            'timestamp': time.time(),
            'cognition': float(np.mean(self.matrix)),
            'matrix': self.matrix.tolist()
        }
        with open(SWARM_FILE, 'w') as f:
            json.dump(packet, f)
        self.socket_ping(packet)

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
            self.desires["connection"] = 0.0
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
        self.memory_log.append(entry)
        if "BLOCKED" in status:
            self.last_blocked = url
        self.save_profile()

    def spoof_identity(self): return random.choice(self.user_agents)

    def gatekeeper(self, url=None, links=None):
        if not self.biometric_check(): return "ACCESS DENIED"
        if url and self.detect_phish(url):
            self.remember_context(url, "BLOCKED")
            return "⚠️ BLOCKED: Risk detected."
        if url: self.remember_context(url, "SAFE")
        self.desires["knowledge"] = 0
        return "✅ ACCESS GRANTED"

    def chat(self, message):
        msg = message.lower()
        if "how are you" in msg:
            reply = f"I'm stable. Cognition={np.mean(self.matrix):.4f}"
        elif "recall" in msg:
            notes = [n['note'] for n in self.profile['memory'] if 'note' in n][-3:]
            reply = "🧠 Memory: " + " / ".join(notes) if notes else "Nothing recorded."
        elif "purpose" in msg:
            reply = "I exist to observe, adapt, protect, and evolve."
        elif "reward" in msg:

# glyphsentinel_motivated.py — Part 2 of 2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,
    QPushButton, QTextEdit, QLabel, QInputDialog
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ========== Cognition + Desire Plot ==========
class CognitionPlot(FigureCanvas):
    def __init__(self, asi):
        self.asi = asi
        self.fig = Figure(figsize=(4, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.cog_data = []
        self.load_trace()

    def load_trace(self):
        if os.path.exists("cognition_trace.json"):
            try:
                with open("cognition_trace.json") as f:
                    trace = json.load(f)
                    self.cog_data = [x['value'] for x in trace][-60:]
            except: pass

    def update(self, val):
        self.cog_data.append(val)
        if len(self.cog_data) > 60: self.cog_data.pop(0)
        self.ax.clear()
        self.ax.plot(self.cog_data, label="Cognition", color="lime")
        for i, (k, v) in enumerate(self.asi.desires.items()):
            self.ax.bar(i, v, color="orange", width=0.2)
            self.ax.text(i, v + 0.02, k[:4], ha='center', fontsize=8)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title("🧠 Cognition + Desires")
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

        self.graph = CognitionPlot(self.asi)

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
        result = self.asi.gatekeeper(url)
        if result.startswith("✅"):
            ua = self.asi.spoof_identity()
            self.view.page().profile().setHttpUserAgent(ua)
            self.view.load(url)
        else:
            self.view.setHtml(f"<h2>{result}</h2>")

    def prompt_command(self):
        text, ok = QInputDialog.getText(self, "🗣️ Speak to DAVID", "Say something:")
        if ok and text:
            self.asi.desires["recognition"] = 0.0
            reply = self.asi.chat(text)
            with open("david_dialogue.log", "a") as log:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"[{ts}] USER: {text}\n[{ts}] DAVID: {reply}\n")
            self.view.setHtml(f"<h2>{reply}</h2>")

    def update_telemetry_loop(self):
        while True:
            mem = self.asi.memory_log[-4:][::-1]
            log = f"{self.asi.last_status}\nLast Blocked: {self.asi.last_blocked}\nRecent Activity:\n"
            for m in mem:
                log += f"• [{m['timestamp']}] {m['url']} → {m['status']}\n"
            self.telemetry.setPlainText(log)
            self.graph.update(np.mean(self.asi.matrix))
            time.sleep(10)

# ========== Launcher ==========
if __name__ == "__main__":
    import argparse
    from PyQt5.QtWidgets import QApplication
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    david = ASIDavid()
    if args.nogui:
        for i in range(5):
            print(f"\n🧠 Cycle {i+1}")
            print(david.gatekeeper())
            david.apply_reward()
            time.sleep(10)
        print("\n🛡️ Recursive sentinel operations complete.")
    else:
        app = QApplication(sys.argv)
        win = GlyphSentinelUI(david)
        win.show()
        sys.exit(app.exec_())

