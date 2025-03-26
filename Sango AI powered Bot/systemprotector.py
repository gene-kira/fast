import ast
import socket
import concurrent.futures
import psutil
import pyshark
import pandas as pd
from sklearn.ensemble import IsolationForest
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class SystemProtector:
    def __init__(self, port_range=(1024, 65535)):
        self.port_range = port_range
        self.open_ports = set()
        self.rogue_programs = set()
        self.network_traffic = []
        self.if_model = IsolationForest(contamination=0.01)
        self.file_watcher = None

    def initialize(self):
        # Initialize network capture
        self.capture_network_traffic()

        # Initialize system resource monitoring
        self.monitor_system_resources()

        # Initialize file system monitoring
        self.monitor_file_changes()

        # Train the anomaly detection model
        self.train_anomaly_detection_model()

    def capture_network_traffic(self):
        capture = pyshark.LiveCapture(interface='eth0')
        capture.apply_on_packets(self.process_packet, packets_count=100)

    def process_packet(self, packet):
        try:
            src_port = int(packet.tcp.srcport)
            dst_port = int(packet.tcp.dstport)
            self.network_traffic.append((src_port, dst_port))
        except AttributeError:
            pass  # Non-TCP packets

    def monitor_system_resources(self):
        psutil.cpu_percent(interval=1)
        for proc in psutil.process_iter(['pid', 'name']):
            if self.is_rogue_program(proc):
                self.rogue_programs.add(proc.pid)

    def is_rogue_program(self, process):
        # Define criteria to identify rogue programs
        return process.name().startswith('malicious') or process.cpu_percent() > 50

    def monitor_file_changes(self):
        class FileChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not os.path.isdir(event.src_path):
                    self.check_file(event.src_path)

            def check_file(self, file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    if 'malicious' in content:
                        print(f"Rogue file detected: {file_path}")

        self.file_watcher = Observer()
        self.file_watcher.schedule(FileChangeHandler(), path='/', recursive=True)
        self.file_watcher.start()

    def train_anomaly_detection_model(self):
        # Collect system resource data
        cpu_usage = []
        mem_usage = []
        for _ in range(100):  # Collect 100 samples
            cpu_usage.append(psutil.cpu_percent(interval=0.1))
            mem_usage.append(psutil.virtual_memory().percent)

        # Create a DataFrame and train the model
        data = pd.DataFrame({'cpu': cpu_usage, 'mem': mem_usage})
        self.if_model.fit(data)

    def detect_anomalies(self):
        # Continuously monitor system resources for anomalies
        while True:
            current_data = {'cpu': [psutil.cpu_percent(interval=0.1)], 'mem': [psutil.virtual_memory().percent]}
            prediction = self.if_model.predict(pd.DataFrame(current_data))
            if prediction[0] == -1:  # Anomaly detected
                print("Anomaly detected in system resources")

    def manage_ports(self):
        for port in range(self.port_range[0], self.port_range[1]):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(('localhost', port)) == 0:
                        self.open_ports.add(port)
            except Exception as e:
                print(f"Error checking port {port}: {e}")

        # Close all open ports
        for port in self.open_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('localhost', port))
                    s.close()
                    print(f"Closed port {port}")
            except Exception as e:
                print(f"Error closing port {port}: {e}")

    def run(self):
        self.initialize()

if __name__ == "__main__":
    protector = SystemProtector()
    protector.run()
