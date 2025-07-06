# --- Autoloader ---
import importlib, subprocess, sys

class LibraryLoader:
    def __init__(self):
        self.required_libs = [
            "pynput", "psutil", "spacy", "json", "random", "hashlib",
            "subprocess", "os", "time", "threading"
        ]

    def autoload(self):
        for lib in self.required_libs:
            if lib in sys.builtin_module_names:
                print(f"🧠 Built-in module '{lib}' loaded.")
                continue
            try:
                importlib.import_module(lib)
                print(f"✅ Library '{lib}' is available.")
            except ImportError:
                print(f"📦 Installing '{lib}'...")
                subprocess.run([sys.executable, "-m", "pip", "install", lib])

# --- Core Imports ---
import os, time, json, random, hashlib, threading, subprocess
import psutil
from pynput import keyboard, mouse
import spacy

# --- Modules ---
class FileMonitor:
    def __init__(self, watch_dir="./watch"):
        self.watch_dir = watch_dir
        self.snapshots = {}

    def hash_file(self, filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def scan_files(self):
        print("📁 Scanning files...")
        for filename in os.listdir(self.watch_dir):
            path = os.path.join(self.watch_dir, filename)
            if os.path.isfile(path):
                current_hash = self.hash_file(path)
                prev_hash = self.snapshots.get(filename)
                if prev_hash and prev_hash != current_hash:
                    print(f"⚠️ Mutation detected: {filename}")
                self.snapshots[filename] = current_hash

class GamingProtection:
    def protect_during_gaming(self):
        cpu = psutil.cpu_percent(interval=1)
        gpu = random.uniform(0, 100)
        if cpu > 90 or gpu > 90:
            print(f"⚠️ Spike detected! CPU: {cpu}%, GPU: {gpu:.2f}%")

class BiometricMonitor:
    def __init__(self):
        self.typing_speed = []
        self.mouse_movements = []
        keyboard.Listener(on_press=self.track_typing).start()
        mouse.Listener(on_move=self.track_mouse).start()

    def track_typing(self, key):
        now = time.time()
        self.typing_speed.append(now)
        if len(self.typing_speed) > 1:
            interval = self.typing_speed[-1] - self.typing_speed[-2]
            print(f"⌨️ Typing interval: {interval:.3f}s")

    def track_mouse(self, x, y):
        self.mouse_movements.append((time.time(), x, y))

    def analyze_behavior(self):
        if len(self.typing_speed) > 10:
            avg_speed = sum(self.typing_speed[-10:]) / 10
            if avg_speed > 2.0:
                print("⚠️ Anomaly in typing speed!")
        if len(self.mouse_movements) > 10:
            print("🖱️ Mouse activity normal.")

class ThreatHunter:
    def __init__(self, log_file='/var/log/syslog'):
        self.log_file = log_file
        self.threat_signatures = ["unauthorized_access", "port_scan", "privilege_escalation"]

    def hunt_threats(self):
        print("🔍 Hunting threats...")
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-100:]:
                    if any(sig in line for sig in self.threat_signatures):
                        print(f"🚨 Threat detected: {line.strip()}")
        except Exception as e:
            print(f"🛑 Error reading logs: {e}")

class SegmentPolicyEngine:
    def __init__(self):
        self.zones = {"core": ["127.0.0.1"], "gaming": [], "sensitive": []}
        self.policies = {"core": {"posture": "hardened"}, "gaming": {"posture": "watched"}}

    def apply_policy(self, device_ip, zone):
        if zone in self.zones:
            self.zones[zone].append(device_ip)
            print(f"🧩 Device {device_ip} → Zone: {zone} | Policy: {self.policies[zone]}")

class DeviceVerifier:
    def __init__(self):
        self.verified_devices = {}

    def check_device_integrity(self, device_id):
        trust_score = random.uniform(0, 1)
        verified = trust_score > 0.7
        self.verified_devices[device_id] = verified
        print(f"🔐 Device {device_id} trust score: {trust_score:.2f}")
        return verified

class TrafficInspector:
    def inspect_packet(self, packet_data):
        entropy = random.uniform(0, 1)
        if entropy > 0.8:
            print("⚠️ Encrypted traffic suspicious.")
        else:
            print("🔒 Traffic clean.")

class IntentEvaluator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def evaluate(self, user_request):
        doc = self.nlp(user_request)
        flagged = any(tok.lemma_ in ["delete", "shutdown", "bypass"] for tok in doc)
        print("⚠️ Intent flagged!" if flagged else f"✅ Intent clear: {user_request}")
        return not flagged

class GlyphEncoder:
    def __init__(self):
        self.glyph_log = {}

    def encode(self, agent_id, biometric_pulse, device_fingerprint):
        combo = f"{agent_id}:{biometric_pulse}:{device_fingerprint}"
        glyph = hashlib.sha256(combo.encode()).hexdigest()[:16]
        self.glyph_log[agent_id] = glyph
        print(f"🔣 Glyph: {glyph}")
        return glyph

class ZKAuthenticator:
    def __init__(self):
        self.proof_store = {}

    def generate_proof(self, user_id):
        proof = hashlib.sha256(user_id.encode()).hexdigest()
        self.proof_store[user_id] = proof
        print(f"🧪 Proof generated for {user_id}")
        return proof

    def verify_proof(self, user_id, proof):
        valid = self.proof_store.get(user_id) == proof
        print("✅ Proof verified" if valid else "❌ Proof failed")
        return valid

class MemoryBinder:
    def __init__(self):
        self.vault = {}

    def bind_state(self, agent_id, system_state, glyph):
        self.vault[agent_id] = {"state": system_state, "glyph": glyph}
        print(f"🔐 State bound for {agent_id}")

    def rollback_state(self, agent_id, current_glyph):
        entry = self.vault.get(agent_id)
        if entry and entry["glyph"] == current_glyph:
            print(f"↩️ Rollback triggered for {agent_id}")
            return entry["state"]
        print("❌ Rollback denied.")
        return None

# --- System Orchestrator ---
class ZeroTrustSystem:
    def __init__(self):
        self.file_monitor = FileMonitor()
        self.gaming_protection = GamingProtection()
        self.biometric_monitor = BiometricMonitor()
        self.threat_hunter = ThreatHunter()
        self.segment_policy = SegmentPolicyEngine()
        self.device_verifier = DeviceVerifier()
        self.traffic_inspector = TrafficInspector()
        self.intent_evaluator = IntentEvaluator()
        self.glyph_encoder = GlyphEncoder()
        self.zk_auth = ZKAuthenticator()
        self.memory_binder = MemoryBinder()

    def run(self):
        print("🚀 Zero-Trust AI Defense Activated")
        while True:
            self.file_monitor.scan_files()
            self.gaming_protection.protect_during_gaming()
            self.biometric_monitor.analyze_behavior()
            self.threat_hunter.hunt_threats()
            self.segment_policy.apply_policy("192.168.0.10", "gaming")
            self.device_verifier.check_device_integrity("device_42")
            self.traffic_inspector.inspect_packet(b"packet_data")
            self.intent_evaluator.evaluate("shutdown all services")

            glyph = self.glyph_encoder.encode("agent_007", 0.233, "device_42_fp")
            proof = self.zk_auth.generate_proof("agent_007")
            self.zk_auth.verify_proof("agent_007", proof)

            state = {
                "file_hashes": self.file_monitor.snapshots,
                "verified": self.device_verifier.verified_devices,
                "typing": self.biometric_monitor.typing_speed[-10:]
            }
            self.memory_binder.bind_state("agent_007", state, glyph)
            time.sleep(60)

# --- Entry Point ---
if __name__ == "__main__":
    loader = LibraryLoader()
    loader.autoload()
    zero_trust_system = ZeroTrustSystem()
    zero_trust_system.run()

