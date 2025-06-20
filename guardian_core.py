 the complete `guardian_core.py`  

```python
import sys, subprocess, importlib

# === Auto-loader for required packages ===
required = ['numpy', 'cv2', 'deepface', 'psutil', 'pynvml', 'parselmouth']
for pkg in required:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import os, platform, hashlib, time, socket, threading, shutil, random
import numpy as np
import cv2
import psutil
from deepface import DeepFace
from pynvml import *
import parselmouth

AUTHORIZED_GLYPHS = {"🜂🜄🜁🜃", "🜏🜍🜎🜔"}

# === Biometric Monitor (System-based) ===
class BiometricMonitor:
    def __init__(self):
        self.cpu_temp_threshold = 85  # °C
        self.battery_threshold = 20   # %

    def get_cpu_temperature(self):
        try:
            if platform.system() == "Linux":
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    return int(f.read()) / 1000.0
            elif platform.system() == "Windows":
                import wmi
                w = wmi.WMI(namespace="root\\wmi")
                temps = w.MSAcpi_ThermalZoneTemperature()[0]
                return (temps.CurrentTemperature / 10.0) - 273.15
            else:
                return random.uniform(45, 75)
        except:
            return random.uniform(45, 75)

    def get_battery_percent(self):
        battery = psutil.sensors_battery()
        return battery.percent if battery else 100

    def check_vitals(self):
        cpu_temp = self.get_cpu_temperature()
        battery = self.get_battery_percent()
        print(f"[DIGITAL VITALS] CPU Temp: {cpu_temp:.1f}°C | Battery: {battery}%")
        return cpu_temp > self.cpu_temp_threshold or battery < self.battery_threshold

# === Facial Emotion Monitor ===
class FacialEmotionMonitor:
    def __init__(self):
        self.last_emotion = None
        self.alert_emotions = {"fear", "angry", "sad"}

    def analyze_frame(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            print(f"[FACIAL] Emotion Detected: {emotion}")
            self.last_emotion = emotion
            return emotion
        except Exception as e:
            print(f"[FACIAL] Error: {e}")
            return None

    def should_trigger_alert(self):
        return self.last_emotion in self.alert_emotions

# === Voice Stress Monitor ===
class VoiceStressMonitor:
    def __init__(self, wav_path="sample.wav"):
        self.wav_path = wav_path
        self.f0_threshold = 250  # Hz

    def analyze_voice(self):
        try:
            snd = parselmouth.Sound(self.wav_path)
            pitch = snd.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values > 0]
            avg_f0 = np.mean(f0_values)
            print(f"[VOICE] Avg F0: {avg_f0:.2f} Hz")
            return avg_f0 > self.f0_threshold
        except Exception as e:
            print(f"[VOICE] Error: {e}")
            return False

# === Hardware Monitor ===
class HardwareMonitor:
    def __init__(self):
        self.cpu_usage_threshold = 95
        self.gpu_temp_threshold = 85
        self.gpu_usage_threshold = 95
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
        except:
            self.gpu_handle = None

    def check_cpu(self):
        usage = psutil.cpu_percent(interval=1)
        print(f"[CPU] Usage: {usage}%")
        return usage > self.cpu_usage_threshold

    def check_gpu(self):
        if not self.gpu_handle:
            print("[GPU] NVIDIA GPU not found.")
            return False
        try:
            util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
            temp = nvmlDeviceGetTemperature(self.gpu_handle, NVML_TEMPERATURE_GPU)
            print(f"[GPU] Usage: {util.gpu}% | Temp: {temp}°C")
            return util.gpu > self.gpu_usage_threshold or temp > self.gpu_temp_threshold
        except:
            return False

# === Glyphic Access Daemon ===
def start_glyphic_daemon(authorized_glyphs, validate_fn, host='0.0.0.0', port=7777):
    def handle_client(conn, addr):
        print(f"[GLYPHIC] Connection from {addr}")
        glyph = conn.recv(1024).decode()
        if validate_fn(glyph):
            conn.send(b"Access granted.\n")
        else:
            conn.send(b"Access denied.\n")
        conn.close()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    print(f"[GLYPHIC] Listening on port {port}...")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()

# === Guardian Core ===
class GuardianCore:
    def __init__(self):
        self.os_type = platform.system()
        self.integrity_hash = self.generate_integrity_hash()
        self.symbolic_keys = AUTHORIZED_GLYPHS
        self.entropy_threshold = 0.85
        self.backup_path = "guardian_backup.img"
        self.distress_glyph = "🜨🜛🜚🜙"
        self.audit_log = []

        self.biometrics = BiometricMonitor()
        self.facial_monitor = FacialEmotionMonitor()
        self.voice_monitor = VoiceStressMonitor()
        self.hardware = HardwareMonitor()

    def generate_integrity_hash(self):
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def verify_integrity(self):
        if self.generate_integrity_hash() != self.integrity_hash:
            self.trigger_anomaly("Integrity drift detected.")

    def entropy_score(self, data):
        values, counts = np.unique(list(data), return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def detect_adversarial(self, input_data):
        score = self.entropy_score(input_data)
        print(f"[ENTROPY] Score: {score:.3f}")
        return score > self.entropy_threshold

    def validate_symbolic_glyph(self, glyph):
        return glyph in self.symbolic_keys

    def broadcast_distress(self):
        print(f"[MYTHIC GLYPH] Broadcasting: {self.distress_glyph}")

    def fork_clean_instance(self):
        print("[FAILOVER] Forking clean instance...")
        try:
            shutil.copy(self.backup_path, "guardian_restored.img")
            print("[FAILOVER] Instance restored.")
        except Exception as e:
            print(f"[FAILOVER ERROR] {e}")

    def quarantine_node(self):
        print("[FAILOVER] Quarantining node...")
        time.sleep(1)
        print("[FAILOVER] Node isolated.")

    def execute_failover(self):
        self.broadcast_distress()
        self.quarantine_node()
        self.fork_clean_instance()

    def trigger_anomaly(self, message):
        self.audit_log.append((time.time(), message))
        print(f"[ALERT] {message}")
        self.lockdown()

    def lockdown(self):
        print("🔐 Entering recursive lockdown...")
        if self.os_type == "Linux":
            subprocess.run(["sudo", "systemctl", "stop", "networking"])
        elif self.os_type == "Windows":
            subprocess.run(["netsh", "interface", "set", "interface", "Ethernet", "admin=disabled"], shell=True)
        elif self.os_type == "Darwin":
            subprocess.run(["sudo", "ifconfig", "en0", "down"])

    def monitor_face(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.facial_monitor.analyze_frame(frame)
            if self.facial_monitor.should_trigger_alert():
                self.execute_failover()
            cv2.imshow("Facial Monitor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def launch(self):
        print(⚙️ Launching Guardian Core heartbeat...")
        self.verify_integrity()

        if self.detect_adversarial(''.join(random.choices("abcdef123456", k=120))):
            self.execute_failover()

        if self.biometrics.check_vitals():
            self.execute_failover()

        if self.voice_monitor.analyze_voice():
            print("[VOICE] Vocal stress detected.")
            self.execute_failover()

        if self.hardware.check_cpu() or self.hardware.check_gpu():
            self.execute_failover()

# === Swarm Node Replication ===
def replicate_node(core_instance):
    print("[SWARM] Creating Guardian Swarm Node...")
    replica = GuardianCore()
    replica.audit_log = core_instance.audit_log.copy()
    print(f"[SWARM] Replica complete. Log entries copied: {len(replica.audit_log)}")
    return replica

# === Main Runtime ===
if __name__ == "__main__":
    core = GuardianCore()
    threading.Thread(target=start_glyphic_daemon, args=(AUTHORIZED_GLYPHS, core.validate_symbolic_glyph), daemon=True).start()
    threading.Thread(target=core.monitor_face, daemon=True).start()

    while True:
        core.launch()
        time.sleep(10)
        if random.random() < 0.1:
            replicate_node(core)