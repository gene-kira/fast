import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import requests
import socket
import subprocess
from datetime import datetime

# Load and process data from files
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file type")

# Anomaly Detection using Isolation Forest
def detect_anomalies(data):
    clf = IsolationForest(contamination=0.01)
    data = data.select_dtypes(include=[np.number])
    clf.fit(data)
    anomalies = clf.predict(data)
    return anomalies

# Behavioral Monitoring
def monitor_behavior(log_file, threshold=5):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    counts = {}
    for line in lines:
        if "action" in line:
            action = line.split(":")[1].strip()
            counts[action] = counts.get(action, 0) + 1
    for action, count in counts.items():
        if count > threshold:
            print(f"High frequency of action: {action} (count: {count})")

# System Integration and Security
def block_ip(ip_address):
    os.system(f"iptables -A INPUT -s {ip_address} -j DROP")

def open_socket(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', port))
    s.listen(5)
    print(f"Socket listening on port {port}")
    return s

# Monitor and protect system files
def monitor_system_files(critical_files):
    for file in critical_files:
        if not os.path.exists(file):
            print(f"Critical file missing: {file}")
            continue
        try:
            with open(file, 'rb') as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            if current_hash != stored_hashes.get(file, ""):
                print(f"File changed: {file} (old hash: {stored_hashes.get(file)}, new hash: {current_hash})")
                stored_hashes[file] = current_hash
        except Exception as e:
            print(f"Failed to monitor {file}: {e}")

# Protect core processes
def protect_core_processes(processes):
    for process in processes:
        try:
            result = subprocess.run(['pgrep', process], capture_output=True, text=True)
            if not result.stdout:
                print(f"Process {process} is not running. Attempting to restart...")
                subprocess.run(['systemctl', 'restart', process])
        except Exception as e:
            print(f"Failed to protect process {process}: {e}")

# Secure file permissions
def secure_file_permissions(critical_files, user='root'):
    for file in critical_files:
        try:
            os.chown(file, 0, 0)  # Set owner and group to root (UID=0, GID=0)
            os.chmod(file, 0o644)  # Set permissions to -rw-r--r--
        except Exception as e:
            print(f"Failed to secure file permissions for {file}: {e}")

# Main AI Agent Function
def main():
    # Load and preprocess data
    data_files = [
        "AI/anomaly-detection-process-and-file-system.py",
        "AI/behavior-monitoring.py",
        "AI/fake_info_interceptor-3-.py",
        "AI/Ghost in the Machine-2-.py",
        "AI/Holodeck Real World Simulation.py",
        "AI/preprocess data.py"
    ]
    
    critical_files = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/group",
        "/etc/hosts",
        "/var/log/auth.log"
    ]

    core_processes = ["sshd", "systemd", "cron"]

    # Initialize stored hashes for critical files
    global stored_hashes
    stored_hashes = {}
    for file in critical_files:
        if os.path.exists(file):
            with open(file, 'rb') as f:
                stored_hashes[file] = hashlib.sha256(f.read()).hexdigest()

    data = {}
    for file in data_files:
        try:
            d = load_data(file)
            data[file] = d
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    # Anomaly Detection
    anomalies = detect_anomalies(data["AI/anomaly-detection-process-and-file-system.py"])
    if -1 in anomalies:
        print("Anomalies detected in system data")

    # Behavioral Monitoring
    monitor_behavior("AI/behavior-monitoring.log")

    # Security Measures
    suspicious_ips = ["192.168.1.10", "10.0.0.5"]
    for ip in suspicious_ips:
        block_ip(ip)

    # Open Sockets for Communication
    ports = [8080, 8081]
    sockets = []
    for port in ports:
        s = open_socket(port)
        sockets.append(s)

    # Continuous Monitoring and Protection
    while True:
        time.sleep(60)  # Check every minute

        # Re-check critical files
        monitor_system_files(critical_files)

        # Re-check core processes
        protect_core_processes(core_processes)

        # Re-check behavior
        monitor_behavior("AI/behavior-monitoring.log")

if __name__ == "__main__":
    main()
