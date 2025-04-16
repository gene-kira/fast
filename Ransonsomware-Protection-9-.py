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
import json
import asyncio
import aiohttp
import base64
import winreg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration file path
CONFIG_FILE = 'config.json'

# Load configuration
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        logging.error("Configuration file not found.")
        sys.exit(1)

config = load_config()

# List of required libraries
required_libraries = [
    'psutil',
    'requests',
    'pynput',
    'pygetwindow',
    'aiohttp',
    'watchdog'
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
async def is_malicious_program(program_path, pid):
    # Example heuristic: check for known malicious filenames or behaviors
    if 'medusa' in os.path.basename(program_path).lower():
        return True

    # Behavior Analysis
    if any(await is_suspicious_behavior(conn) for conn in psutil.net_connections() if conn.pid == pid):
        return True

    # Signature Matching
    if await is_malicious_signature(program_path):
        return True

    # File System Monitoring
    if await is_ransomware_behavior(program_path, pid):
        return True

    # Real-Time Threat Intelligence
    program_hash = hashlib.md5(open(program_path, 'rb').read()).hexdigest()
    if await check_file_reputation(program_hash):
        logging.warning(f"Malicious file detected: {program_path}")
        return True

    for conn in psutil.net_connections():
        if conn.pid == pid:
            ip = conn.raddr.ip
            if await check_ip_reputation(ip):
                logging.warning(f"Suspicious IP connection detected: {ip} from program {program_path}")
                return True

    # Memory Monitoring for Fileless Malware
    if await is_fileless_malware(program_path, pid):
        return True

    return False

# Function to check for suspicious behavior patterns
async def is_suspicious_behavior(connection):
    # Example: check for connections to known malicious IP addresses or ports
    if connection.raddr.ip in ['192.168.1.100', '10.0.0.1']:
        return True
    if connection.raddr.port in [6881, 6882, 6883]:
        return True
    return False

# Function to check for known malicious signatures
async def is_malicious_signature(program_path):
    # Example: check the process's hash against a list of known hashes
    program_hash = hashlib.md5(open(program_path, 'rb').read()).hexdigest()
    with open('malicious_hashes.json', 'r') as f:
        malicious_hashes = json.load(f)
    if program_hash in malicious_hashes:
        return True
    return False

# Function to detect file encryption behavior
def monitor_file_changes(dir):
    class FileChangeHandler(FileSystemEventHandler):
        def __init__(self):
            self.file_changes = {}

        def on_modified(self, event):
            dir = os.path.dirname(event.src_path)
            if dir in self.file_changes:
                current_files = set(os.listdir(dir))
                if len(current_files - self.file_changes[dir]) > 5:
                    logging.warning(f"Rapid file changes detected in {dir}")
                    terminate_process(os.getpid())
            else:
                self.file_changes[dir] = set(os.listdir(dir))

    handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(handler, dir, recursive=True)
    observer.start()

# Function to check registry modifications
def monitor_registry():
    def on_registry_change(key):
        try:
            value, _ = winreg.QueryValueEx(key, 'Medusa')
            if value == '1':
                logging.warning("Medusa Ransomware detected in registry")
                terminate_process(os.getpid())
        except FileNotFoundError:
            pass

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows\CurrentVersion\Run', 0, winreg.KEY_ALL_ACCESS)
    winreg.SetValueEx(key, 'Medusa', 0, winreg.REG_SZ, '1')
    winreg.CloseKey(key)

# Function to detect and prevent security software disabling
def monitor_security_software():
    def on_security_change(process):
        if process.name() in ['antivirus.exe', 'firewall.exe']:
            logging.warning("Security software disabled by Medusa Ransomware")
            subprocess.run(['taskkill', '/F', '/IM', process.name()])
            subprocess.run(['sc', 'start', process.name()])

    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] in ['antivirus.exe', 'firewall.exe']:
            monitor_security_change(proc)

# Function to handle mouse clicks
def on_click(x, y, button, pressed):
    active_window = get_active_window_title()
    pid = get_pid_from_window_title(active_window)
    program_path = get_program_path(pid)
    if program_path and is_malicious_program(program_path, pid):
        logging.warning(f"Medusa Ransomware detected: {program_path}")
        terminate_process(pid)

# Function to terminate a process
def terminate_process(pid):
    try:
        psutil.Process(pid).terminate()
        logging.info(f"Process terminated: {pid}")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logging.error(f"Failed to terminate process {pid}: {e}")

# Main function
async def main():
    with MouseListener(on_click=on_click) as listener:
        while True:
            # Continuously monitor running processes
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    pid = proc.info['pid']
                    program_path = get_program_path(pid)
                    if is_malicious_program(program_path, pid):
                        logging.warning(f"Medusa Ransomware detected: {program_path}")
                        terminate_process(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logging.error(f"Error monitoring process {pid}: {e}")

            # Monitor file changes in user directories
            for dir in ['C:\\Users', 'D:\\Documents']:
                monitor_file_changes(dir)

            # Monitor registry modifications
            monitor_registry()

            # Monitor security software disabling
            monitor_security_software()

            await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
