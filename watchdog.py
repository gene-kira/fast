import nmap
import json
import numpy as np
from sklearn.ensemble import IsolationForest
import subprocess
import time
import os

# Ensure necessary libraries are installed
try:
    import nmap
    import json
    import numpy as np
    from sklearn.ensemble import IsolationForest
    import subprocess
    import time
    import os
except ImportError as e:
    print(f"Missing required library: {e}")
    exit(1)

def scan_ports(target_ip):
    nm = nmap.PortScanner()
    nm.scan(hosts=target_ip, arguments='-p 0-65535')
    open_ports = []
    
    for host in nm.all_hosts():
        if 'tcp' in nm[host]:
            for port in nm[host]['tcp']:
                open_ports.append(port)
    
    return open_ports

def collect_baseline_data(target_ip, duration=60):
    all_open_ports = []
    
    for _ in range(duration // 5):  # Collect data every 5 seconds for the specified duration
        open_ports = scan_ports(target_ip)
        all_open_ports.append(open_ports)
        time.sleep(5)
    
    with open('baseline_data.json', 'w') as f:
        json.dump(all_open_ports, f)

def train_anomaly_detector(baseline_data):
    flat_data = [item for sublist in baseline_data for item in sublist]
    unique_ports = list(set(flat_data))
    
    X = []
    for ports in baseline_data:
        row = [1 if port in ports else 0 for port in unique_ports]
        X.append(row)
    
    X = np.array(X)
    
    model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
    model.fit(X)
    
    return model, unique_ports

def detect_and_terminate_anomalies(open_ports, model, unique_ports):
    X = [1 if port in open_ports else 0 for port in unique_ports]
    X = np.array([X])
    
    prediction = model.predict(X)
    
    if prediction[0] == -1:  # Anomaly detected
        print("Anomaly Detected! Open Ports:", open_ports)
        
        for port in open_ports:
            try:
                result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
                pids = [line.split()[1] for line in result.stdout.splitlines()[1:]]
                for pid in pids:
                    subprocess.run(['kill', '-9', pid])
                    print(f"Terminated process with PID {pid} on port {port}")
            except Exception as e:
                print(f"Failed to terminate processes on port {port}: {e}")

def mock_evaluation_function(model, unique_ports):
    # Generate synthetic data for evaluation
    normal_data = [[1 if i in [22, 80, 443] else 0 for i in unique_ports]]
    anomaly_data = [[1 if i in [22, 80, 443, 2100, 3000] else 0 for i in unique_ports]]

    # Evaluate normal data
    normal_predictions = model.predict(normal_data)
    print("Normal Data Prediction:", "Anomaly" if normal_predictions[0] == -1 else "No Anomaly")

    # Evaluate anomaly data
    anomaly_predictions = model.predict(anomaly_data)
    print("Anomaly Data Prediction:", "Anomaly" if anomaly_predictions[0] == -1 else "No Anomaly")

def main(target_ip="192.168.1.1", duration=60):
    if not os.path.exists('baseline_data.json'):
        collect_baseline_data(target_ip, duration)  # Collect data for 60 seconds if baseline doesn't exist

    with open('baseline_data.json', 'r') as f:
        baseline_data = json.load(f)

    model, unique_ports = train_anomaly_detector(baseline_data)
    
    # Mock evaluation
    mock_evaluation_function(model, unique_ports)
    
    while True:
        open_ports = scan_ports(target_ip)
        detect_and_terminate_anomalies(open_ports, model, unique_ports)
        time.sleep(60)  # Check every 60 seconds

if __name__ == "__main__":
    main()
