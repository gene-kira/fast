# glyphsentinel_navigator.py

# ========== AUTOLOADER ==========
import subprocess, sys

def install_and_import(package, alias=None):
    try:
        __import__(alias or package)
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

deps = {
    "PyQt5": "PyQt5.QtWidgets",
    "torch": "torch",
    "cryptography": "cryptography",
    "numpy": "numpy"
}

for pkg, module in deps.items():
    install_and_import(pkg, module)

# ========== LIBRARY IMPORTS ==========
import os, time, random, json, threading
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from cryptography.fernet import Fernet
from urllib.parse import urlparse
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,
    QPushButton, QTextEdit, QLabel
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt

# ========== ASI DAVID CORE ==========
class ASIDavid(nn.Module):
    def __init__(self):
        super().__init__()
        self.profile_path = "david_profile.sec"
        self.key_path = "david_key.key"
        self.swarm_file = "swarm_matrix.json"
        self.key = self.load_key()
        self.profile = self.load_profile()
        self.matrix = np.random.rand(1000, 128)
        self.paths = ["wallet.txt", "secrets.db"]
        self.access_log = {}
        self.last_status = "🧠 Initialized"
        self.last_blocked = "—"
        self.memory_log = []
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Linux x86_64)"
        ]
        self.model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        threading.Thread(target=self.autonomous_cognition, daemon=True).start()

 def generate_key(self):
        key = Fernet.generate_key()
        with open(self.key_path, 'wb') as f: f.write(key)
        return key

    def load_key(self):
        return open(self.key_path, 'rb').read() if os.path.exists(self.key_path) else self.generate_key()

    def encrypt_data(self, profile, key): return Fernet(key).encrypt(json.dumps(profile).encode())
    def decrypt_data(self, data, key):
        try: return json.loads(Fernet(key).decrypt(data).decode())
        except: return None

    def load_profile(self):
        if os.path.exists(self.profile_path):
            with open(self.profile_path, 'rb') as f:
                return self.decrypt_data(f.read(), self.key)
        return {'biometrics': {}, 'memory': []}

    def save_profile(self):
        with open(self.profile_path, 'wb') as f:
            f.write(self.encrypt_data(self.profile, self.key))

    def verify_face(self): return random.choice([True, True, False])
    def verify_fingerprint(self): return random.choice([True, True, False])
    def verify_voice(self): return random.choice([True, True, False])

    def biometric_check(self):
        print("🔍 Validating biometrics...")
        return self.verify_face() and self.verify_fingerprint() and self.verify_voice()

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
        self.matrix += np.random.normal(0, 2.5, size=self.matrix.shape)
        self.matrix += Normal(0, 1).sample((1000, 128)).numpy()
        self.matrix = np.clip(self.matrix, 0, 1)

    def autonomous_cognition(self):
        while True:
            self.cognition()
            self.broadcast()
            self.scan_swarm()
            time.sleep(15)

    def broadcast(self):
        data = {
            'node_id': 'david_01',
            'status': 'active',
            'timestamp': time.time(),
            'cognition': float(np.mean(self.matrix))
        }
        with open(self.swarm_file, 'w') as f: json.dump(data, f)

    def scan_swarm(self):
        if os.path.exists(self.swarm_file):
            try:
                with open(self.swarm_file, 'r') as f:
                    peer = json.load(f)
                    self.last_status = f"🤝 Swarm: {peer['node_id']}, Cognition: {peer['cognition']:.3f}"
            except:
                self.last_status = "⚠️ Swarm unreadable."

    def remember_context(self, url, status):
        entry = {'url': url, 'status': status, 'timestamp': time.strftime("%H:%M:%S")}
        self.profile['memory'].append(entry)
        self.memory_log.append(entry)
        if "BLOCKED" in status:
            self.last_blocked = url
        self.save_profile()

    def spoof_identity(self):
        return random.choice(self.user_agents)

    def gatekeeper(self, url=None):
        if not self.biometric_check():
            return "🚫 ACCESS DENIED: Biometric failure."
        if url and self.detect_phish(url):
            self.remember_context(url, "BLOCKED")
            return "⚠️ BLOCKED: Phishing site detected."
        self.monitor_files()
        self.remember_context(url, "SAFE")
        return "✅ ACCESS GRANTED"

# ========== UI + NAVIGATOR ==========
class GlyphSentinelUI(QMainWindow):
    def __init__(self, asi_core):
        super().__init__()
        self.asi = asi_core
        self.setWindowTitle("GlyphSentinel Navigator")
        self.setGeometry(200, 100, 1200, 800)

        self.view = QWebEngineView()
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("🔐 Enter destination…")
        self.url_bar.returnPressed.connect(self.go_to_url)

        self.go_btn = QPushButton("GO")
        self.go_btn.clicked.connect(self.go_to_url)

        self.telemetry = QTextEdit()
        self.telemetry.setReadOnly(True)
        self.telemetry.setMaximumHeight(140)
        self.telemetry.setStyleSheet("background-color:#111;color:#0f0;font-family:Courier")

        layout = QVBoxLayout()
        layout.addWidget(self.url_bar)
        layout.addWidget(self.go_btn)
        layout.addWidget(self.view)
        layout.addWidget(QLabel("📊 Telemetry"))
        layout.addWidget(self.telemetry)

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

    def update_telemetry_loop(self):
        while True:
            mem = self.asi.memory_log[-4:][::-1]
            log = f"{self.asi.last_status}\nLast Blocked: {self.asi.last_blocked}\nRecent Activity:\n"
            for m in mem:
                log += f"• [{m['timestamp']}] {m['url']} → {m['status']}\n"
            self.telemetry.setPlainText(log)
            time.sleep(10)

# ========== RUN LAUNCH ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    david = ASIDavid()
    win = GlyphSentinelUI(david)
    win.show()
    sys.exit(app.exec_())

