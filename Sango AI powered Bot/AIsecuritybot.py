import os
import time
import requests
import psutil
import pyshark
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.ensemble import RandomForestClassifier
import pickle
import threading

# Define the main AI bot class
class AISecurityBot:
    def __init__(self):
        self.port_monitor = PortMonitor()
        self.activity_scanner = ActivityScanner()
        self.rogue_detector = RogueDetector()
        self.memory_scanner = MemoryScanner()
        self.response_system = ResponseSystem()
        self.machine_learning = MachineLearningEngine()
        self.load_model()

    def load_model(self):
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.train_model()

    def train_model(self):
        # Collect training data
        known_threats = requests.get('https://threatintelapi.com/threats').json()
        normal_activities = self.activity_scanner.collect_normal_activities()

        # Prepare the dataset
        X, y = [], []
        for threat in known_threats:
            features = extract_features(threat)
            X.append(features)
            y.append(1)  # Threat

        for activity in normal_activities:
            features = extract_features(activity)
            X.append(features)
            y.append(0)  # Normal

        self.model = RandomForestClassifier()
        self.model.fit(X, y)
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def start(self):
        threading.Thread(target=self.port_monitor.start).start()
        threading.Thread(target=self.activity_scanner.start).start()
        threading.Thread(target=self.rogue_detector.start).start()
        threading.Thread(target=self.memory_scanner.start).start()
        threading.Thread(target=self.response_system.start).start()
        self.machine_learning.start()

# Port Management Module
class PortMonitor:
    def __init__(self):
        self.open_ports = set()
        self.closed_ports = set()

    def start(self):
        while True:
            current_ports = {p.laddr.port for p in psutil.process_iter(['laddr'])}
            new_open_ports = current_ports - self.open_ports
            closed_ports = self.open_ports - current_ports

            if new_open_ports:
                print(f"New ports opened: {new_open_ports}")
                # Check and handle new open ports
                for port in new_open_ports:
                    self.handle_new_port(port)

            if closed_ports:
                print(f"Ports closed: {closed_ports}")
                # Handle closed ports
                for port in closed_ports:
                    self.handle_closed_port(port)

            self.open_ports = current_ports
            time.sleep(5)  # Check every 5 seconds

    def handle_new_port(self, port):
        if not self.is_legitimate(port):
            print(f"Port {port} is suspicious. Closing it.")
            self.close_port(port)
        else:
            self.open_ports.add(port)

    def handle_closed_port(self, port):
        if port in self.closed_ports:
            print(f"Port {port} re-opened. Checking legitimacy.")
            if not self.is_legitimate(port):
                self.close_port(port)
            else:
                self.open_ports.add(port)

    def is_legitimate(self, port):
        # Use machine learning to determine legitimacy
        features = extract_features({'port': port})
        return self.model.predict([features])[0] == 0

    def close_port(self, port):
        os.system(f"sudo iptables -A INPUT -p tcp --dport {port} -j DROP")

# Real-Time Port Activity Scanner
class ActivityScanner:
    def __init__(self):
        self.captured = pyshark.LiveCapture(interface='eth0')  # Change to your network interface

    def collect_normal_activities(self):
        # Collect a dataset of normal activities for training
        normal_activities = []
        for packet in self.captured.sniff_continuously(packet_count=1000):
            if 'TCP' in packet:
                activity = {
                    'src_ip': packet.ip.src,
                    'dst_ip': packet.ip.dst,
                    'src_port': packet.tcp.srcport,
                    'dst_port': packet.tcp.dstport
                }
                normal_activities.append(activity)
        return normal_activities

    def start(self):
        while True:
            for packet in self.captured.sniff_continuously(packet_count=100):
                if 'TCP' in packet:
                    activity = {
                        'src_ip': packet.ip.src,
                        'dst_ip': packet.ip.dst,
                        'src_port': packet.tcp.srcport,
                        'dst_port': packet.tcp.dstport
                    }
                    self.check_activity(activity)

    def check_activity(self, activity):
        features = extract_features(activity)
        if self.model.predict([features])[0] == 1:
            print(f"Anomalous activity detected: {activity}")
            # Handle the anomalous activity (e.g., log it and trigger response system)

# Rogue Program Detector
class RogueDetector:
    def __init__(self):
        self.rogue_programs = set()
        self.known_signatures = requests.get('https://threatintelapi.com/signatures').json()

    def start(self):
        while True:
            for process in psutil.process_iter(['name', 'exe']):
                if self.is_rogue(process):
                    print(f"Rogue program detected: {process}")
                    self.handle_rogue_program(process)

    def is_rogue(self, process):
        # Use machine learning to determine legitimacy
        features = extract_features({'process_name': process.name, 'process_exe': process.exe})
        return self.model.predict([features])[0] == 1 or process.name in self.known_signatures

    def handle_rogue_program(self, process):
        try:
            process.terminate()
            print(f"Process {process} terminated.")
        except psutil.NoSuchProcess:
            pass
        finally:
            if os.path.exists(process.exe):
                os.remove(process.exe)
                print(f"File {process.exe} deleted.")

# System Memory Scanner
class MemoryScanner:
    def __init__(self):
        self.rogue_memory = set()

    def start(self):
        while True:
            for process in psutil.process_iter(['memory_info']):
                if self.is_rogue_memory(process):
                    print(f"Rogue memory detected: {process}")
                    self.handle_rogue_memory(process)

    def is_rogue_memory(self, process):
        # Use machine learning to determine legitimacy
        features = extract_features({'process_name': process.name, 'memory_info': process.memory_info})
        return self.model.predict([features])[0] == 1

    def handle_rogue_memory(self, process):
        try:
            process.terminate()
            print(f"Process {process} terminated.")
        except psutil.NoSuchProcess:
            pass

# Response System
class ResponseSystem:
    def start(self):
        while True:
            self.isolate_threats()
            self.terminate_threats()
            self.delete_files()
            time.sleep(60)  # Check every minute

    def isolate_threats(self):
        for port in self.ai_bot.port_monitor.closed_ports:
            if not self.ai_bot.port_monitor.is_legitimate(port):
                print(f"Isolating port {port}")
                os.system(f"sudo iptables -A INPUT -p tcp --dport {port} -j DROP")

    def terminate_threats(self):
        for process in self.ai_bot.rogue_detector.rogue_programs:
            try:
                process.terminate()
                print(f"Process {process} terminated.")
            except psutil.NoSuchProcess:
                pass

    def delete_files(self):
        for file_path in [p.exe for p in self.ai_bot.rogue_detector.rogue_programs]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted.")

# Machine Learning Engine
class MachineLearningEngine:
    def start(self):
        threading.Thread(target=self.update_threat_database).start()
        threading.Thread(target=self.train_model_continuously).start()

    def update_threat_database(self):
        while True:
            try:
                new_threats = requests.get('https://threatintelapi.com/threats').json()
                with open('known_threats.pkl', 'wb') as f:
                    pickle.dump(new_threats, f)
                self.train_model()
            except Exception as e:
                print(f"Error updating threat database: {e}")
            time.sleep(3600)  # Update every hour

    def train_model_continuously(self):
        while True:
            try:
                known_threats = requests.get('https://threatintelapi.com/threats').json()
                normal_activities = self.ai_bot.activity_scanner.collect_normal_activities()

                X, y = [], []
                for threat in known_threats:
                    features = extract_features(threat)
                    X.append(features)
                    y.append(1)  # Threat

                for activity in normal_activities:
                    features = extract_features(activity)
                    X.append(features)
                    y.append(0)  # Normal

                self.ai_bot.model.fit(X, y)
                with open('model.pkl', 'wb') as f:
                    pickle.dump(self.ai_bot.model, f)

                print("Model retrained successfully.")
            except Exception as e:
                print(f"Error training model: {e}")
            time.sleep(3600)  # Retrain every hour

# Feature Extraction
def extract_features(data):
    features = []
    if 'port' in data:
        features.append(data['port'])
    if 'process_name' in data:
        features.append(len(data['process_name']))
    if 'src_ip' in data and 'dst_ip' in data:
        features.append(int(ipaddress.ip_address(data['src_ip'])))
        features.append(int(ipaddress.ip_address(data['dst_ip'])))
    if 'src_port' in data and 'dst_port' in data:
        features.append(data['src_port'])
        features.append(data['dst_port'])
    if 'memory_info' in data:
        features.extend([data['memory_info'].rss, data['memory_info'].vms])

    return features

if __name__ == "__main__":
    ai_bot = AISecurityBot()
    ai_bot.start()
