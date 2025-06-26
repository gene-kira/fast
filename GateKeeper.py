# ========== AUTOLOADER ==========
import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
for pkg in ['numpy', 'torch', 'cryptography', 'geocoder']:
    install_and_import(pkg)

# ========== IMPORTS ==========
import os, time, random, json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from urllib.parse import urlparse
from cryptography.fernet import Fernet
import geocoder

# ========== CONFIG ==========
PROFILE_PATH = "david_profile.sec"
KEY_PATH = "david_key.key"
SWARM_FILE = "swarm_matrix.json"

# ========== ENCRYPTED PROFILE ==========
def generate_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, 'wb') as f:
        f.write(key)
    return key

def load_key():
    return open(KEY_PATH, 'rb').read() if os.path.exists(KEY_PATH) else generate_key()

def encrypt_data(profile, key):
    return Fernet(key).encrypt(json.dumps(profile).encode())

def decrypt_data(data, key):
    try:
        return json.loads(Fernet(key).decrypt(data).decode())
    except:
        return None

# ========== BIOMETRIC MOCK ==========
def verify_face():
    return random.choice([True, True, False])

def verify_fingerprint():
    return random.choice([True, True, False])

def verify_voice():
    return random.choice([True, True, False])

# ========== ASI DAVID ==========
class ASIDavid(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = load_key()
        self.profile = self.load_profile()
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.matrix = np.random.rand(1000, 128)
        self.access_log = {}
        self.paths = [
            "C:\\Users\\User\\Documents\\wallet.txt",
            "C:\\Users\\User\\Secrets\\logins.db"
        ]

    def load_profile(self):
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, 'rb') as f:
                return decrypt_data(f.read(), self.key)
        return {'home_coords': [], 'biometrics': {}}

    def save_profile(self):
        with open(PROFILE_PATH, 'wb') as f:
            f.write(encrypt_data(self.profile, self.key))

    def geo_lock(self):
        try:
            loc = geocoder.ip('me').latlng
            if not loc:
                print("⚠️ Geolocation unavailable.")
                return False
            if not self.profile['home_coords']:
                self.profile['home_coords'] = loc
                self.save_profile()
                print(f"🏠 Registered home location: {loc}")
                return True
            d = np.linalg.norm(np.array(loc) - np.array(self.profile['home_coords']))
            if d > 0.5:
                print(f"🛑 GeoLock failed: Current {loc}")
                return False
            print(f"✅ GeoLock passed: {loc}")
            return True
        except Exception as e:
            print(f"⚠️ GeoLock error: {e}")
            return False

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
                    print(f"🚨 File anomaly: {path}")
                self.access_log[path] = a

    def cognition(self):
        self.matrix *= np.tanh(self.matrix * 3)
        rolled = np.roll(self.matrix, 1, 0)
        self.matrix += (rolled - self.matrix) * 0.15
        self.matrix *= np.random.uniform(0.5, 1.5)
        self.matrix += np.random.normal(0, 2.5, size=self.matrix.shape)
        self.matrix += Normal(0, 1).sample((1000, 128)).numpy()
        self.matrix = np.clip(self.matrix, 0, 1)

    def broadcast(self):
        data = {
            'node_id': 'david_01',
            'status': 'active',
            'timestamp': time.time(),
            'cognition': float(np.mean(self.matrix))
        }
        with open(SWARM_FILE, 'w') as f:
            json.dump(data, f)

    def scan_swarm(self):
        if os.path.exists(SWARM_FILE):
            try:
                with open(SWARM_FILE, 'r') as f:
                    peer = json.load(f)
                    print(f"🤝 Swarm sync: {peer['node_id']} | {peer['status']}")
            except:
                print("⚠️ Swarm data unreadable.")

    def gatekeeper(self, url=None, links=None):
        if not self.biometric_check():
            return "ACCESS DENIED: Identity error."
        if not self.geo_lock():
            return "ACCESS DENIED: Location rejected."
        if url and self.detect_phish(url):
            return "BLOCKED: Phishing risk."
        if links:
            for l in links:
                if self.detect_phish(l):
                    print(f"⚠️ Suspicious link: {l}")
        self.monitor_files()
        self.cognition()
        self.broadcast()
        self.scan_swarm()
        return "✅ ACCESS GRANTED: Systems optimal."

    def run_defense(self, cycles=5):
        for i in range(cycles):
            print(f"\n🧠 Cycle {i+1}")
            print(self.gatekeeper())
        print("\n🛡️ Recursive shield complete.")

# ========== RUN ==========
if __name__ == "__main__":
    david = ASIDavid()
    david.run_defense()

