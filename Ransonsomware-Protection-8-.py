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
    'aiohttp'
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
    if 'malicious' in os.path.basename(program_path).lower():
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
    # Example: check the process's hash against a database of known malware hashes
    program_hash = hashlib.md5(open(program_path, 'rb').read()).hexdigest()
    if program_hash in await load_malware_database():
        return True
    return False

# Function to load a database of known malware hashes
async def load_malware_database():
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
async def is_ransomware_behavior(program_path, pid):
    # Example: check for rapid file modifications in critical directories
    monitored_dirs = ['C:\\Users', 'D:\\Documents']
    file_changes = {}

    async def monitor_files(dir):
        while True:
            try:
                current_files = set(os.listdir(dir))
                if dir in file_changes and len(current_files - file_changes[dir]) > 5:
                    logging.warning(f"Rapid file changes detected in {dir}")
                    return True
                file_changes[dir] = current_files
            except Exception as e:
                logging.error(f"Error monitoring files in {dir}: {e}")
            await asyncio.sleep(1)

    tasks = [asyncio.create_task(monitor_files(dir)) for dir in monitored_dirs]
    done, pending = await asyncio.wait(tasks)
    return any(task.result() for task in done)

# Function to check IP reputation using AbuseIPDB
async def check_ip_reputation(ip):
    url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&key={config['abuseipdb_api_key']}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            if data['data']['abuseConfidenceScore'] > 50:
                return True
    return False

# Function to check file reputation using VirusTotal
async def check_file_reputation(file_hash):
    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {"x-apikey": config['virustotal_api_key']}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            data = await response.json()
            if data['data']['attributes']['last_analysis_stats']['malicious'] > 0:
                return True
    return False

# Function to detect fileless malware in memory
async def is_fileless_malware(program_path, pid):
    try:
        process = psutil.Process(pid)
        for mem_info in process.memory_maps():
            if 'powershell' in mem_info.path.lower() or 'cmd' in mem_info.path.lower():
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False

# Function to decode obfuscated code
def decode_obfuscated_code(obfuscated_code):
    try:
        decoded = base64.b64decode(obfuscated_code).decode('utf-8')
        if 'malicious' in decoded.lower():
            return True
    except Exception as e:
        logging.error(f"Error decoding obfuscated code: {e}")
    return False

# Function to handle mouse clicks
def on_click(x, y, button, pressed):
    active_window = get_active_window_title()
    pid = get_pid_from_window_title(active_window)
    program_path = get_program_path(pid)
    if program_path and is_malicious_program(program_path, pid):
        logging.warning(f"Malware detected: {program_path}")
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
                        logging.warning(f"Malware detected: {program_path}")
                        terminate_process(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logging.error(f"Error monitoring process {pid}: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
