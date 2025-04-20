import psutil
import socket
from scapy.all import sniff, IP, TCP
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from gtts import gTTS
import smtplib
from email.mime.text import MIMEText
import os
import time
from threading import Thread
from sklearn.ensemble import IsolationForest
import numpy as np

# Global variables for known entities
known_ips = set([ip for ip in psutil.net_if_addrs().values()])
known_devices = set(os.listdir('/dev'))
known_processes = set()
known_files = set()

def collect_baseline_data(duration=3600):
    cpu_usage = []
    memory_usage = []
    network_traffic = []

    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory().percent
        net_io = psutil.net_io_counters()

        cpu_usage.append(cpu_percent)
        memory_usage.append(memory_info)
        network_traffic.append(net_io.bytes_sent + net_io.bytes_recv)

    return np.column_stack((cpu_usage, memory_usage, network_traffic))

def train_anomaly_detector(data):
    model = IsolationForest(contamination=0.01)
    model.fit(data)
    return model

def detect_anomalies(model, data):
    anomalies = model.predict(data)
    return [data[i] for i in range(len(anomalies)) if anomalies[i] == -1]

def send_voice_alert(message):
    tts = gTTS(text=message, lang='en')
    tts.save('alert.mp3')
    os.system("mpg123 alert.mp3")

def send_email_alert(email, subject, message):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = 'your_email@example.com'
    msg['To'] = email

    with smtplib.SMTP('smtp.example.com') as server:
        server.login('your_email@example.com', 'your_password')
        server.sendmail('your_email@example.com', [email], msg.as_string())

def trigger_alert(message, ip=None, program_name=None, filename=None):
    alert_message = f"Alert: {message}"
    if ip:
        alert_message += f", IP: {ip}"
    if program_name:
        alert_message += f", Program: {program_name}"
    if filename:
        alert_message += f", File: {filename}"

    send_voice_alert(alert_message)
    send_email_alert('admin@example.com', 'System Alert', alert_message)

def block_process(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()
        print(f"Process {pid} terminated")
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")

def quarantine_file(file_path, quarantine_dir='/quarantine'):
    if not os.path.exists(quarantine_dir):
        os.makedirs(quarantine_dir)
    shutil.move(file_path, os.path.join(quarantine_dir, os.path.basename(file_path)))
    print(f"File {file_path} quarantined")

def block_network_connection(ip):
    with open('/etc/hosts.deny', 'a') as f:
        f.write(f"{ip}\n")
    print(f"Network connection to {ip} blocked")

class SystemMonitor:
    def __init__(self):
        self.process_monitor = ProcessMonitor()
        self.file_monitor = FileMonitor()
        self.network_monitor = NetworkMonitor()

    def start(self):
        threads = [
            Thread(target=self.monitor_hardware),
            Thread(target=self.process_monitor.monitor_processes),
            Thread(target=self.monitor_file_system),
            Thread(target=self.network_monitor.monitor_network)
        ]

        for thread in threads:
            thread.start()

        while True:
            self.check_anomalies()
            time.sleep(10)

    def monitor_hardware(self):
        known_devices = set(os.listdir('/dev'))
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            devices = os.listdir('/dev')

            if cpu_percent > 75 or memory_info.percent > 80:
                trigger_alert("High CPU or Memory usage detected", ip=None, program_name=None, filename=None)

            sniff(filter="ip", prn=lambda x: process_network_packet(x))

            for device in devices:
                if 'usb' in device and device not in known_devices:
                    trigger_alert(f"New USB device detected: {device}", ip=None, program_name=device, filename=None)
                    known_devices.add(device)

    def monitor_file_system(self):
        observer = Observer()
        handler = self.file_monitor
        observer.schedule(handler, path='/', recursive=True)
        observer.start()

    def check_anomalies(self):
        current_data = collect_baseline_data(duration=60)
        anomalies = detect_anomalies(anomaly_detector, current_data)

        for anomaly in anomalies:
            if anomaly[0] > 75 or anomaly[1] > 80:
                trigger_alert(f"High resource usage detected: CPU {anomaly[0]}, Memory {anomaly[1]}", ip=None, program_name=None, filename=None)
            
            if anomaly[2] > 100000000:  # Large network traffic
                trigger_alert("Large network traffic detected", ip=None, program_name=None, filename=None)

        for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
            cpu_percent = proc.info['cpu_percent']
            memory_info = proc.info['memory_info']
            
            if cpu_percent > 50 or memory_info.percent > 50:
                trigger_alert(f"High resource usage detected: {proc.info}", ip=None, program_name=proc.info['name'], filename=None)
                block_process(proc.pid)

        for conn in psutil.net_connections():
            src_ip = conn.laddr
            dst_ip = conn.raddr
            
            if src_ip not in known_ips or dst_ip not in known_ips:
                trigger_alert(f"Unusual network traffic detected: {src_ip} -> {dst_ip}", ip=dst_ip, program_name=None, filename=None)
                block_network_connection(dst_ip)

class ProcessMonitor:
    def __init__(self):
        self.known_processes = set()

    def monitor_processes(self):
        while True:
            current_processes = set([p.pid for p in psutil.process_iter()])
            new_processes = current_processes - self.known_processes
            if new_processes:
                print(f"New processes detected: {new_processes}")
            
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
                cpu_percent = proc.info['cpu_percent']
                memory_info = proc.info['memory_info']
                
                if cpu_percent > 50 or memory_info.percent > 50:
                    trigger_alert(f"High resource usage detected: {proc.info}", ip=None, program_name=proc.info['name'], filename=None)
                    block_process(proc.pid)

            self.known_processes = current_processes

class FileMonitor(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            known_files.add(file_path)
            trigger_alert(f"File modified: {file_path}", ip=None, program_name=None, filename=file_path)

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            known_files.add(file_path)
            trigger_alert(f"New file created: {file_path}", ip=None, program_name=None, filename=file_path)

class NetworkMonitor:
    def monitor_network(self):
        while True:
            for conn in psutil.net_connections():
                src_ip = conn.laddr
                dst_ip = conn.raddr
                
                if src_ip not in known_ips or dst_ip not in known_ips:
                    trigger_alert(f"Unusual network traffic detected: {src_ip} -> {dst_ip}", ip=dst_ip, program_name=None, filename=None)
                    block_network_connection(dst_ip)

if __name__ == "__main__":
    # Collect baseline data
    baseline_data = collect_baseline_data(duration=3600)
    anomaly_detector = train_anomaly_detector(baseline_data)

    monitor = SystemMonitor()
    monitor.start()
