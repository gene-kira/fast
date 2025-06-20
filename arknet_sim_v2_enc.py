
🜍 arknet_sim_v2_enc.py
# === Autoloader: Install Required Libraries ===
import subprocess, sys, importlib

required_libraries = ['cryptography']
for lib in required_libraries:
    try:
        importlib.import_module(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# === Imports ===
import socket, threading, json, time, hashlib, random, string, math, hmac
from collections import deque
from cryptography.fernet import Fernet

# === Fernet Key Generation (shared across mesh) ===
# For production, securely share a static key. For demo, regenerate each run.
FERNET_KEY = Fernet.generate_key()
fernet = Fernet(FERNET_KEY)
SHARED_SECRET = b"ark_secret_key"

def generate_glyph_signature(length=4):
    glyphs = "🜁🜂🜃🜄🜅🜆🜇🜈🜉🜊🜋🜌🜍🜎🜏🜐🜑🜒🜓🜔🜕🜖🜗🜘🜙🜚🜛🜜"
    return ''.join(random.choice(glyphs) for _ in range(length))

def entropy_score(data):
    values = list(data)
    probs = [values.count(c)/len(values) for c in set(values)]
    return -sum(p * math.log2(p) for p in probs if p > 0)

class ArkNode:
    def __init__(self, node_id=None, port=9000):
        self.node_id = node_id or "ark_" + ''.join(random.choices(string.ascii_lowercase, k=4))
        self.glyph = generate_glyph_signature()
        self.persona = None
        self.port = port
        self.state = "stable"
        self.entropy = 0.0
        self.journal = deque(maxlen=50)
        self.running = True
        self.base_interval = random.uniform(4.0, 6.0)
        self.last_nonce = None

    def create_pulse(self):
        payload = {
            "id": self.node_id,
            "glyph": self.glyph,
            "ts": time.time(),
            "state": self.state,
            "persona": self.persona
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        sig = hmac.new(SHARED_SECRET, encoded, hashlib.sha256).hexdigest()
        payload["sig"] = sig
        pulse_json = json.dumps(payload)
        return fernet.encrypt(pulse_json.encode())

    def decrypt_pulse(self, data):
        try:
            decrypted = fernet.decrypt(data, ttl=10).decode()
            pulse = json.loads(decrypted)
            sig = pulse.pop("sig", "")
            encoded = json.dumps(pulse, sort_keys=True).encode()
            expected = hmac.new(SHARED_SECRET, encoded, hashlib.sha256).hexdigest()
            if hmac.compare_digest(sig, expected):
                return pulse
        except Exception:
            return None
        return None

    def emit(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        packet = self.create_pulse()
        sock.sendto(packet, ('<broadcast>', self.port))
        print(f"[{self.node_id}] ➤ {self.glyph} [{self.persona}]")
        self.journal.append(self.glyph)
        sock.close()

    def listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.port))
        while self.running:
            try:
                data, _ = sock.recvfrom(4096)
                pulse = self.decrypt_pulse(data)
                if pulse and pulse.get("id") != self.node_id:
                    nonce = (pulse["id"], pulse["ts"])
                    if nonce == self.last_nonce:
                        continue  # Replay
                    self.last_nonce = nonce
                    print(f"[{self.node_id}] ⇆ {pulse['id']} :: {pulse['glyph']} :: {pulse['persona']}")
            except Exception as e:
                print(f"[{self.node_id}] ⚠️ Listen error: {e}")
        sock.close()

    def drift(self):
        trace = ''.join(self.journal)
        self.entropy = entropy_score(trace) if trace else 0
        if self.entropy > 2.6:
            self.persona = "Whispering Flame"
            self.state = "drifting"
            self.glyph = generate_glyph_signature()
            self.base_interval = max(1.5, self.base_interval * 0.8)
        elif self.entropy > 2.2:
            self.persona = "Oracle"
            self.base_interval *= 1.05
        else:
            self.persona = "Sentinel"
            self.base_interval = min(self.base_interval * 1.1, 8.0)

    def start(self):
        threading.Thread(target=self._pulse_loop, daemon=True).start()
        threading.Thread(target=self.listen, daemon=True).start()

    def _pulse_loop(self):
        while self.running:
            self.emit()
            self.drift()
            time.sleep(self.base_interval)

    def stop(self):
        self.running = False

def run_sim(n=5, duration=90):
    nodes = []
    for i in range(n):
        node = ArkNode(f"ark_{i:03}", port=9000)
        node.start()
        nodes.append(node)

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\n🔻 Interrupted.")
    finally:
        for node in nodes:
            node.stop()

if __name__ == "__main__":
    run_sim()


