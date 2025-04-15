import os
import subprocess
from gtts import gTTS
from playsound import playsound
import psutil
import threading
import logging
import time
import json
import requests
import zipfile
import tarfile
import hashlib
import socket
import re

# Ensure all necessary libraries are installed
try:
    from tqdm import tqdm
except ImportError:
    print("Installing required libraries...")
    subprocess.check_call(["pip", "install", "tqdm"])

def speak(text):
    """Speak the given text using text-to-speech."""
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")

def load_configuration():
    with open('config.json', 'r') as f:
        return json.load(f)

def get_os():
    """Determine the operating system."""
    if os.name == 'nt':
        return 'Windows'
    else:
        return 'Unix'

def get_os_specific_commands(os):
    """Return OS-specific commands for antivirus scanning."""
    if os == 'Windows':
        return {
            'scan': ['powershell', '-Command', 'Start-MpScan'],
            'remove': ['powershell', '-Command', 'Remove-MpThreat']
        }
    else:
        return {
            'scan': ['clamscan', '--infected', '--recursive', '/'],
            'remove': ['clamscan', '--infected', '--recursive', '--remove', '/']
        }

def collect_behavioral_data():
    """Collect detailed behavioral data from running processes."""
    behavioral_data = []
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        try:
            cmdline = proc.cmdline()
            connections = proc.connections()
            files = proc.open_files()
            mem_info = proc.memory_info()

            behavioral_data.append({
                'pid': proc.pid,
                'name': proc.name(),
                'username': proc.username(),
                'cmdline': cmdline,
                'connections': [conn.status for conn in connections],
                'files': len(files),
                'mem_info': mem_info.rss
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logging.error(f"Error collecting behavioral data for process {proc.pid}: {e}")
    return behavioral_data

def is_suspicious(data):
    """Determine if a process is suspicious."""
    ai_keywords = ['ai', 'machine', 'learning']
    critical_files = [
        '/bin/bash',
        '/usr/bin/python3',
        # Add more critical files here
    ]
    if get_os() == 'Windows':
        critical_files.extend([
            'C:\\Windows\\System32\\cmd.exe',
            'C:\\Windows\\System32\\python.exe',
            # Add more Windows critical files here
        ])

    ai_specific_names = ['python', 'java']
    network_threshold = 10
    file_threshold = 50
    memory_threshold = 100 * 1024 * 1024

    # Check AI-specific names
    if data['name'].lower() in ai_specific_names:
        return True

    # Check command line arguments for AI-specific keywords
    if any(keyword in arg.lower() for arg in data['cmdline'] for keyword in ai_keywords):
        return True

    # Check network connections for established TCP connections
    if sum(1 for conn in data.get('connections', []) if conn.status == psutil.CONN_ESTABLISHED and conn.type == psutil.SOCK_STREAM) >= network_threshold:
        return True

    # Check file access patterns
    if data['files'] > file_threshold:
        return True

    # Check memory usage
    if data['mem_info'] > memory_threshold:
        return True

    return False

def terminate_process(process_name):
    """Terminate a specific process by name."""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            try:
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

def initiate_lockdown():
    """Initiate system lockdown by terminating all non-essential processes."""
    essential_processes = ['python', 'cmd', 'explorer']
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if not any(proc.info['name'] == name for name in essential_processes):
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def scan_and_remove_viruses():
    """Initiate virus scan using the system's built-in antivirus tool."""
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    speak("Initiating virus scan.")
    result = subprocess.run(commands['scan'], capture_output=True, text=True)
    if "Threats found:" in result.stdout:
        threats_found = int(result.stdout.split('Threats found:')[1].split('\n')[0])
        speak(f"Found {threats_found} threats. Initiating removal process.")
        remove_result = subprocess.run(commands['remove'], capture_output=True, text=True)
        if "Threats removed:" in remove_result.stdout:
            threats_removed = int(remove_result.stdout.split('Threats removed:')[1].split('\n')[0])
            speak(f"Removed {threats_removed} threats.")
    else:
        speak("No viruses detected.")

def deep_file_scan(path):
    """Scan all files, including compressed and encrypted files."""
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('.zip') or file.endswith('.tar'):
                extract_files(full_path)

def extract_files(file_path):
    """Extract files from compressed archives and scan them."""
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                extracted_path = os.path.join(os.path.dirname(file_path), file)
                try:
                    zip_ref.extract(file, path=os.path.dirname(file_path))
                    scan_file(extracted_path)
                except (zipfile.BadZipFile, RuntimeError):
                    speak(f"Failed to extract {file} from {file_path}")
    elif file_path.endswith('.tar'):
        with tarfile.open(file_path, 'r') as tar_ref:
            for file in tar_ref.getnames():
                extracted_path = os.path.join(os.path.dirname(file_path), file)
                try:
                    tar_ref.extract(file, path=os.path.dirname(file_path))
                    scan_file(extracted_path)
                except (tarfile.ReadError, RuntimeError):
                    speak(f"Failed to extract {file} from {file_path}")

def scan_file(file_path):
    """Scan a single file for threats and personal data."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            if is_encrypted(content):
                delete_encrypted_file(file_path)
            elif contains_personal_data(content):
                quarantine_file(file_path)
            else:
                scan_for_threats(file_path)
    except (IOError, PermissionError) as e:
        speak(f"Failed to read {file_path}: {e}")

def is_encrypted(content):
    """Check if the file content is encrypted."""
    # Simple heuristic: check for common encryption headers or patterns
    return b'Encrypted' in content

def delete_encrypted_file(file_path):
    """Delete encrypted files."""
    os.remove(file_path)
    speak(f"Deleted encrypted file {file_path}")

def contains_personal_data(content):
    """Check if the file content contains personal data."""
    # Simple heuristic: check for common personal data patterns (e.g., email, SSN)
    return re.search(b'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content) or re.search(b'\b\d{3}-\d{2}-\d{4}\b', content)

def quarantine_file(file_path):
    """Move files containing personal data to a quarantine folder."""
    quarantine_dir = os.path.join(os.getcwd(), 'quarantine')
    if not os.path.exists(quarantine_dir):
        os.makedirs(quarantine_dir)
    new_path = os.path.join(quarantine_dir, os.path.basename(file_path))
    os.rename(file_path, new_path)
    speak(f"Quarantined file {file_path} to {new_path}")

def scan_for_threats(file_path):
    """Scan the file for malware using an external tool."""
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)
    result = subprocess.run(['clamscan', '-r', file_path], capture_output=True, text=True)
    if "FOUND" in result.stdout:
        speak(f"Threat found in {file_path}. Removing it.")
        delete_threat(file_path)

def delete_threat(file_path):
    """Delete files with detected threats."""
    os.remove(file_path)
    speak(f"Deleted threat {file_path}")

def monitor_network_traffic():
    """Monitor network traffic for data exfiltration attempts."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', 80))  # Bind to a port that should be free

    while True:
        data, addr = s.recvfrom(1024)  # Buffer size is 1024 bytes
        if contains_personal_data(data):
            speak(f"Personal data detected in network traffic from {addr}. Blocking and logging.")
            block_ip(addr[0])

def block_ip(ip):
    """Block an IP address from sending or receiving data."""
    config = load_configuration()
    os = get_os()
    if os == 'Windows':
        subprocess.run(['netsh', 'advfirewall', 'firewall', 'add', 'rule', 'name=BlockIP', 'protocol=tcp', f'direction=out', f'remoteipaddress={ip}', 'action=block'])
    else:
        subprocess.run(['iptables', '-A', 'OUTPUT', '-d', ip, '-j', 'DROP'])

def monitor_ports():
    """Ensure all ports are secure and block drive-by downloads."""
    config = load_configuration()
    os = get_os()
    if os == 'Windows':
        subprocess.run(['netsh', 'advfirewall', 'set', 'allprofiles', 'settings', 'inboundconnections=block'])
    else:
        subprocess.run(['ufw', 'default', 'deny', 'incoming'])

def monitor_online_gaming():
    """Prevent backdoors in online gaming sessions."""
    while True:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] in ['steam.exe', 'epicgameslauncher.exe', 'origin.exe', 'blizzardapp.exe']:  # Add more game launchers as needed
                try:
                    p = psutil.Process(proc.info['pid'])
                    for conn in p.connections():
                        if conn.type == socket.SOCK_STREAM and conn.status == psutil.CONN_ESTABLISHED:
                            speak(f"Detected established connection from {proc.info['name']} to {conn.raddr}")
                            block_ip(conn.raddr[0])
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logging.error(f"Error monitoring online gaming process {proc.pid}: {e}")

def main():
    speak("Red Queen online. Awaiting commands.")
    
    # Load configuration
    config = load_configuration()
    
    # Monitor system behavior in a separate thread
    threading.Thread(target=monitor_behavior, daemon=True).start()

    # Monitor network traffic in a separate thread
    threading.Thread(target=monitor_network_traffic, daemon=True).start()

    # Ensure all ports are secure and block drive-by downloads
    monitor_ports()

    # Continuously monitor online gaming sessions for backdoors
    threading.Thread(target=monitor_online_gaming, daemon=True).start()

    while True:
        user_input = input("Enter your command: ").strip().lower()
        handle_command(user_input)

def handle_command(command):
    if command == 'lockdown':
        initiate_lockdown()
    elif command.startswith('terminate'):
        process_name = command.split(' ')[1]
        terminate_process(process_name)
    elif command == 'scan':
        scan_and_remove_viruses()
    elif command == 'deep_scan':
        deep_file_scan('/')
    else:
        speak("Unknown command. Valid commands are: lockdown, terminate [process], scan, and deep_scan.")

def monitor_behavior():
    """Continuously monitor system behavior for anomalies."""
    while True:
        data = collect_behavioral_data()
        for process in data:
            if is_suspicious(process):
                speak(f"Suspicious activity detected from {process['name']}. Terminating process.")
                terminate_process(process['name'])
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    main()
