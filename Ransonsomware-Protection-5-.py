import os
import psutil
import socket
import hashlib
import logging
import time
from importlib.util import find_spec
import subprocess
import sys
from pynput.mouse import Listener as MouseListener
import tkinter as tk
from tkinter import messagebox
import requests

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of required libraries
required_libraries = [
    'psutil',
    'requests',
    'pynput',
    'pygetwindow'
]

def install_library(library):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])
    except Exception as e:
        logging.error(f"Failed to install {library}: {e}")

# Check and install required libraries
for library in required_libraries:
    if not find_spec(library):
        logging.info(f"{library} not found. Installing...")
        install_library(library)

# Function to check if a program is malicious
def is_malicious_program(program_path):
    # Example heuristic: check for known malicious filenames or behaviors
    if 'malicious' in os.path.basename(program_path).lower():
        return True

    # Behavior Analysis
    if any(is_suspicious_behavior(conn) for conn in psutil.net_connections() if conn.pid == get_pid_from_window_title(get_active_window_title())):
        return True

    # Signature Matching
    if is_malicious_signature(program_path):
        return True

    # File System Monitoring
    if is_ransomware_behavior(program_path):
        return True

    # Real-Time Threat Intelligence
    program_hash = hashlib.md5(open(program_path, 'rb').read()).hexdigest()
    if check_file_reputation(program_hash):
        logging.warning(f"Malicious file detected: {program_path}")
        return True

    for conn in psutil.net_connections():
        if conn.pid == get_pid_from_window_title(get_active_window_title()):
            ip = conn.raddr.ip
            if check_ip_reputation(ip):
                logging.warning(f"Suspicious IP connection detected: {ip} from program {program_path}")
                return True

    return False

# Function to check for suspicious behavior patterns
def is_suspicious_behavior(connection):
    # Example: check for connections to known malicious IP addresses or ports
    if connection.raddr.ip in ['192.168.1.100', '10.0.0.1']:
        return True
    if connection.raddr.port in [6881, 6882, 6883]:
        return True
    return False

# Function to check for known malicious signatures
def is_malicious_signature(program_path):
    # Example: check the process's hash against a database of known malware hashes
    program_hash = hashlib.md5(open(program_path, 'rb').read()).hexdigest()
    if program_hash in load_malware_database():
        return True
    return False

# Function to load a database of known malware hashes
def load_malware_database():
    # Example: read from a file or an API
    with open('malware_hashes.txt', 'r') as f:
        return set(f.read().splitlines())

# Function to get the active window title
def get_active_window_title():
    import pygetwindow as gw
    active_window = gw.getActiveWindow()
    if active_window:
        return active_window.title
    return None

# Function to get the PID of a window by its title
def get_pid_from_window_title(title):
    for process in psutil.process_iter(['pid', 'name']):
        try:
            p = psutil.Process(process.info['pid'])
            if p.name() in title or title in p.name():
                return process.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

# Function to get the program path from a PID
def get_program_path(pid):
    try:
        p = psutil.Process(pid)
        return p.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

# Function to monitor file system changes for ransomware behavior
def is_ransomware_behavior(program_path):
    # Example: check for rapid file modifications in critical directories
    monitored_directories = ['C:\\Users\\', 'D:\\Documents\\']
    file_changes = []

    def monitor_files():
        for dir in monitored_directories:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(file_path)
                        file_changes.append((file_path, mtime))
                    except FileNotFoundError:
                        pass

    monitor_files()
    time.sleep(5)  # Wait for a short period to detect changes
    monitor_files()

    # Check for rapid changes
    if len(file_changes) > 10:  # Threshold for rapid changes
        logging.warning(f"Rapid file modifications detected: {program_path}")
        return True

    return False

# Function to check IP reputation using AbuseIPDB API
def check_ip_reputation(ip):
    url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&key=YOUR_API_KEY"
    response = requests.get(url)
    data = response.json()
    return data['data']['abuseConfidenceScore'] > 50  # Threshold for suspicious IP

# Function to check file reputation using VirusTotal API
def check_file_reputation(file_hash):
    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {"x-apikey": "YOUR_API_KEY"}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data['data']['attributes']['last_analysis_stats']['malicious'] > 0

# Mouse click listener
def on_click(x, y, button, pressed):
    if not pressed:
        active_window_title = get_active_window_title()
        pid = get_pid_from_window_title(active_window_title)
        if pid:
            program_path = get_program_path(pid)
            if is_malicious_program(program_path):
                logging.warning(f"Potentially malicious program detected: {program_path}")
                show_confirmation_dialog(program_path, pid)

# Function to terminate a process
def terminate_process(pid):
    try:
        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=3)  # Wait for the process to terminate
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logging.error(f"Failed to terminate process: {e}")

# Function to show a confirmation dialog
def show_confirmation_dialog(program_path, pid):
    response = messagebox.askyesno("Confirmation", f"Potentially malicious program detected: {program_path}\nDo you really want to run this program?")
    if not response:
        terminate_process(pid)
        logging.info(f"Blocked execution of potentially malicious program: {program_path}")

# Main function
def main():
    # Start mouse click listener
    with MouseListener(on_click=on_click) as listener:
        listener.join()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    main()
