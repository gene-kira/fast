
import os
import requests
import time
import random
import hashlib
import threading
import psutil  # System monitoring
import json
from cryptography.fernet import Fernet
from collections import defaultdict

class AdaptiveAISecurity:
    def __init__(self):
        self.hidden_code_path = "hidden_code.py"
        self.alert_log = "security_alerts.log"
        self.behavior_data = "behavior_learning.json"
        self.lock = threading.Lock()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.expected_hash = None
        self.threat_patterns = defaultdict(int)  # Stores anomaly trends
        self.learning_threshold = 5  # Adjust based on security needs

    def write_code(self):
        """Securely writes AI code while tracking integrity."""
        with self.lock:
            with open(self.hidden_code_path, 'w') as f:
                f.write("def secure_function(): pass")
            self.expected_hash = self.calculate_hash(self.hidden_code_path)
            print(f"Code written to {self.hidden_code_path}")

    def calculate_hash(self, file_path):
        """Generates SHA-256 hash to track file integrity."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def self_preservation(self):
        """Monitors file integrity & restores if tampered."""
        with self.lock:
            current_hash = self.calculate_hash(self.hidden_code_path)
            if current_hash != self.expected_hash:
                print("Warning: Code integrity compromised! Restoring...")
                self.write_code()
                self.trigger_countermeasure("File integrity violation detected!")

    def behavioral_learning(self):
        """Analyzes and adapts to recurring threat patterns."""
        with self.lock:
            try:
                with open(self.behavior_data, 'r') as f:
                    behavior_log = json.load(f)
            except FileNotFoundError:
                behavior_log = {}

            # Monitor security logs & learn threat patterns
            with open(self.alert_log, 'r') as f:
                alerts = f.readlines()

            for alert in alerts:
                if alert.strip():
                    self.threat_patterns[alert.strip()] += 1

            # If a specific threat pattern exceeds the threshold, adjust defense
            for pattern, count in self.threat_patterns.items():
                if count >= self.learning_threshold:
                    print(f"🚨 Adjusting security protocols due to repeated threat: {pattern}")
                    self.trigger_countermeasure(f"Adaptive response triggered for {pattern}")

            # Save learned behaviors
            behavior_log.update(self.threat_patterns)
            with open(self.behavior_data, 'w') as f:
                json.dump(behavior_log, f)

    def proactive_scanning(self):
        """Monitors system activity for anomalies."""
        with self.lock:
            suspicious_processes = ["malicious.exe", "intruder.py"]
            active_processes = [p.name() for p in psutil.process_iter()]
            
            # Detect unauthorized processes
            for process in suspicious_processes:
                if process in active_processes:
                    print(f"⚠️ Suspicious process detected: {process}")
                    self.trigger_countermeasure(f"Unauthorized process {process} running!")

            # Monitor abnormal CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 85:
                print(f"⚠️ High CPU usage detected ({cpu_usage}%)")
                self.trigger_countermeasure(f"Potential system exploitation detected! ({cpu_usage}%) CPU usage")

    def trigger_countermeasure(self, alert_msg):
        """Deploys security countermeasures upon detecting threats."""
        with self.lock:
            self.log_alert(alert_msg)
            print("🛡️ Deploying adaptive security measures...")
            os.system("netsh advfirewall set allprofiles state on")

    def log_alert(self, msg):
        """Logs security incidents for learning."""
        with self.lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.alert_log, 'a') as f:
                f.write(f"[{timestamp}] ALERT: {msg}\n")
            print(f"🚨 Security Alert Logged: {msg}")

    def start(self):
        """Runs AI security agent with adaptive learning."""
        tasks = [
            threading.Thread(target=self.write_code),
            threading.Thread(target=self.self_preservation),
            threading.Thread(target=self.behavioral_learning),
            threading.Thread(target=self.proactive_scanning),
        ]

        for task in tasks:
            task.start()

        for task in tasks:
            task.join()

if __name__ == "__main__":
    ai_security = AdaptiveAISecurity()
    ai_security.start()



