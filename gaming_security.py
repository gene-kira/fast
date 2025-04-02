import os
import socket
import subprocess
import psutil
import logging
from datetime import datetime
import threading
import requests
from scapy.all import sniff, IP, TCP
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set up logging
logging.basicConfig(filename='gaming_security.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(message):
    logging.info(message)
    print(f"[{datetime.now()}] {message}")

def install_libraries():
    """Install all necessary libraries."""
    try:
        os.system('pip3 install scapy tensorflow sklearn pandas')
        log("All necessary libraries installed.")
    except Exception as e:
        log(f"Error installing libraries: {e}")

def monitor_network():
    """Monitor network traffic for suspicious activity using machine learning."""
    def process_packet(packet):
        if IP in packet and TCP in packet:
            try:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                features = [src_ip, dst_ip, src_port, dst_port]
                df.loc[len(df)] = features
            except Exception as e:
                log(f"Error processing packet: {e}")

    def detect_suspicious_traffic(df):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.drop(columns=['src_ip', 'dst_ip']))
        
        model = IsolationForest(contamination=0.1)
        model.fit(X_scaled)
        
        df['anomaly'] = model.predict(X_scaled)
        suspicious_traffic = df[df['anomaly'] == -1]
        if not suspicious_traffic.empty:
            for index, row in suspicious_traffic.iterrows():
                log(f"Suspicious traffic detected: {row}")

    df = pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port'])
    sniff(prn=process_packet)
    detect_suspicious_traffic(df)

def monitor_processes():
    """Monitor running processes for unauthorized activity."""
    known_good_processes = ['python', 'steam', 'discord', 'chrome']  # Add your trusted processes here

    def process_monitor():
        while True:
            try:
                processes = list(psutil.process_iter(['pid', 'name']))
                for proc in processes:
                    if proc.info['name'].lower() not in known_good_processes:
                        log(f"Suspicious process detected: {proc}")
            except Exception as e:
                log(f"Error monitoring processes: {e}")

    threading.Thread(target=process_monitor, daemon=True).start()

def apply_security_patches():
    """Apply security patches and updates."""
    try:
        # Update system
        os.system('sudo apt update && sudo apt upgrade -y')
        log("System updated with latest patches.")
        
        # Update software
        os.system('sudo snap refresh --all')
        log("Software updated with latest patches.")
    except Exception as e:
        log(f"Error applying security patches: {e}")

def scan_for_malware():
    """Scan the system for malware using ClamAV."""
    try:
        # Install ClamAV if not already installed
        os.system('sudo apt install clamav -y')
        
        # Update virus definitions
        os.system('sudo freshclam')
        
        # Scan the system
        scan_result = subprocess.run(['clamscan', '-r', '/'], capture_output=True, text=True)
        log(f"Malware scan completed. Result: {scan_result.stdout}")
    except Exception as e:
        log(f"Error scanning for malware: {e}")

def block_game_launch():
    """Block the game from launching if a threat is detected."""
    def is_threat_detected():
        with open('gaming_security.log', 'r') as file:
            lines = file.readlines()
            for line in lines[::-1]:
                if "Suspicious traffic detected" in line or "Suspicious process detected" in line or "Malware scan completed. Result: FOUND" in line:
                    return True
        return False

    def block_game():
        game_process_name = 'your_game_process_name'  # Replace with the actual name of your game's process
        if is_threat_detected():
            log("Threat detected, blocking game launch.")
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'].lower() == game_process_name:
                    try:
                        os.kill(proc.info['pid'], 9)
                        log(f"Killed process: {proc}")
                    except Exception as e:
                        log(f"Error killing process: {e}")

    threading.Thread(target=block_game, daemon=True).start()

def delete_infected_files():
    """Delete infected files on the local system and other devices on the same network."""
    def delete_local_files(scan_result):
        for line in scan_result.stdout.splitlines():
            if "FOUND" in line:
                file_path = line.split(':')[0]
                try:
                    os.remove(file_path)
                    log(f"Deleted infected file: {file_path}")
                except Exception as e:
                    log(f"Error deleting file: {e}")

    def delete_network_files(ip, scan_result):
        for line in scan_result.stdout.splitlines():
            if "FOUND" in line:
                file_path = line.split(':')[0]
                try:
                    # Assuming the other devices have the same user and permissions
                    os.system(f'ssh {ip} "rm -f {file_path}"')
                    log(f"Deleted infected file on {ip}: {file_path}")
                except Exception as e:
                    log(f"Error deleting file on {ip}: {e}")

    def scan_and_delete():
        # Scan local system
        local_scan_result = subprocess.run(['clamscan', '-r', '/'], capture_output=True, text=True)
        delete_local_files(local_scan_result)

        # Scan other devices on the same network
        ip_network = '.'.join(socket.gethostbyname(socket.gethostname()).split('.')[:-1]) + '.'
        for i in range(1, 255):
            ip = f"{ip_network}{i}"
            if ip != socket.gethostbyname(socket.gethostname()):
                try:
                    scan_result = subprocess.run(['ssh', ip, 'clamscan -r /'], capture_output=True, text=True)
                    delete_network_files(ip, scan_result)
                except Exception as e:
                    log(f"Error scanning and deleting files on {ip}: {e}")

    threading.Thread(target=scan_and_delete, daemon=True).start()

def main():
    install_libraries()
    
    # Start network monitoring
    import threading
    network_thread = threading.Thread(target=monitor_network)
    network_thread.daemon = True
    network_thread.start()
    
    # Start process monitoring
    monitor_processes()
    
    # Apply security patches and updates
    apply_security_patches()
    
    # Scan for malware
    scan_for_malware()
    
    # Block game launch if a threat is detected
    block_game_launch()
    
    # Delete infected files on the local system and other devices on the same network
    delete_infected_files()

if __name__ == "__main__":
    main()
