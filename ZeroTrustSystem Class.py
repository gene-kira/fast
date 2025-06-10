


### Python Script

```python
import os
import hashlib
import socket
import time
from zipfile import ZipFile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.ensemble import IsolationForest
import pandas as pd

# Define the main class for the zero-trust system
class ZeroTrustSystem:
    def __init__(self):
        self.file_monitor = FileMonitor()
        self.port_security = PortSecurity()
        self.anomaly_detector = AnomalyDetector()
        self.gaming_protection = GamingProtection()

    def run(self):
        print("Starting Zero-Trust System...")
        while True:
            self.file_monitor.scan_files()
            self.port_security.check_ports()
            self.anomaly_detector.detect_anomalies()
            self.gaming_protection.protect_during_gaming()
            time.sleep(60)  # Check every minute

class FileMonitor:
    def __init__(self):
        self.sensitive_keywords = ["SSN", "credit card", "password"]
        self.malware_signatures = []  # Load malware signatures from a database or file
        self.isolation_forest = IsolationForest(contamination=0.01)
        self.train_model()

    def train_model(self):
        # Train the model with normal file access patterns
        data = pd.read_csv('normal_access_patterns.csv')
        self.isolation_forest.fit(data)

    def scan_files(self):
        print("Scanning files for malicious content and personal data...")
        for root, dirs, files in os.walk('/path/to/monitor'):
            for file in files:
                file_path = os.path.join(root, file)
                if self.check_file(file_path):
                    print(f"Malicious or sensitive content detected: {file_path}")
                    os.remove(file_path)

    def check_file(self, file_path):
        # Check for malware signatures
        with open(file_path, 'rb') as f:
            file_content = f.read()
            if any(signature in file_content for signature in self.malware_signatures):
                return True

        # Check for personal data
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
            if any(keyword in content for keyword in self.sensitive_keywords):
                return True

        # Check for zip files with password protection
        if file.endswith('.zip'):
            try:
                with ZipFile(file_path, 'r') as z:
                    if z.is_encrypted():
                        os.remove(file_path)
                        print(f"Deleted encrypted zip file: {file_path}")
                        return True
            except Exception as e:
                print(f"Error processing zip file: {e}")

        # Check for encrypted files
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                if self.is_encrypted(content):
                    os.remove(file_path)
                    print(f"Deleted encrypted file: {file_path}")
                    return True
        except Exception as e:
            print(f"Error processing file: {e}")

        # Check for anomalies
        if self.detect_anomaly(file_path):
            return True

        return False

    def is_encrypted(self, content):
        try:
            cipher = Cipher(algorithms.AES(content[:16]), modes.CBC(content[16:32]), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_data = decryptor.update(content[32:]) + decryptor.finalize()
            return False  # If decryption is successful, it's not encrypted
        except:
            return True

    def detect_anomaly(self, file_path):
        # Use the trained model to detect anomalies
        features = self.extract_features(file_path)
        if len(features) == 0:
            return False
        prediction = self.isolation_forest.predict([features])
        return prediction[0] == -1

    def extract_features(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                features = [len(content), hashlib.sha256(content).hexdigest()]
                return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return []

class PortSecurity:
    def check_ports(self):
        print("Checking port security...")
        for port in range(1, 1024):  # Common ports
            if self.is_port_open(port):
                print(f"Port {port} is open and should be closed.")
                self.close_port(port)

    def is_port_open(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0

    def close_port(self, port):
        # Implement logic to close the port
        print(f"Closing port {port}...")

class AnomalyDetector:
    def detect_anomalies(self):
        print("Detecting anomalies in system behavior...")
        # Example: Monitor network traffic and file access patterns
        self.monitor_network_traffic()
        self.monitor_file_access()

    def monitor_network_traffic(self):
        # Implement logic to monitor network traffic for anomalies
        print("Monitoring network traffic...")

    def monitor_file_access(self):
        # Implement logic to monitor file access patterns for anomalies
        print("Monitoring file access patterns...")

class GamingProtection:
    def protect_during_gaming(self):
        print("Protecting system during online gaming...")
        # Example: Block known malicious IPs and ports used in backdoor attacks
        self.block_malicious_ips()

    def block_malicious_ips(self):
        # Implement logic to block known malicious IPs
        print("Blocking malicious IPs...")

if __name__ == "__main__":
    zero_trust_system = ZeroTrustSystem()
    zero_trust_system.run()
```

