import os
import logging
import base64
import hashlib
import schedule
import psutil
import requests
from flask import Flask, jsonify
from flask_limiter import Limiter, RequestLimit
from flask_limiter.util import get_remote_address
from aiohttp import ClientSession
import aioschedule
import asyncio
import win32com.client  # For Office file analysis on Windows

# Configuration loading
def load_configuration():
    with open('security_config.json', 'r') as config_file:
        return json.load(config_file)

config = load_configuration()

# Function to check if a file is an Office document
def is_office_document(file_path):
    office_extensions = ['.docx', '.xlsx', '.pptx']
    return any(file_path.endswith(ext) for ext in office_extensions)

# Function to decode obfuscated code within the document
def decode_obfuscated_code(obfuscated_code):
    try:
        decoded = base64.b64decode(obfuscated_code).decode('utf-8')
        if 'malicious' in decoded.lower():
            return True
    except Exception as e:
        logging.error(f"Error decoding obfuscated code: {e}")
    return False

# Function to extract macros from Office documents
def extract_macros(file_path):
    try:
        word = win32com.client.Dispatch("Word.Application")
        document = word.Documents.Open(file_path)
        vba_project = document.VBProject
        for component in vba_project.VBE.ActiveVBProject.VBComponents:
            if component.Type == 1:  # vbext_StdModule
                macro_code = component.CodeModule.Lines(1, component.CodeModule.CountOfLines)
                if decode_obfuscated_code(macro_code):
                    logging.warning(f"Malicious macro detected in {file_path}")
        document.Close()
        word.Quit()
    except Exception as e:
        logging.error(f"Error extracting macros from {file_path}: {e}")

# Function to monitor Office processes
def monitor_office_processes():
    office_apps = ['word', 'excel', 'powerpoint']

    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'].lower() in office_apps:
            process = psutil.Process(proc.info['pid'])
            try:
                connections = process.connections()
                if any(is_suspicious_behavior(conn) for conn in connections):
                    logging.warning(f"Suspicious behavior detected from {proc.info['name']} with PID {proc.info['pid']}")
                    terminate_process(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

# Function to check for suspicious behavior patterns
async def is_suspicious_behavior(connection):
    if connection.raddr.ip in config['known_malicious_ips']:
        return True
    if connection.status == 'ESTABLISHED' and connection.type == 'TCP':
        ip = connection.raddr.ip
        if await check_ip_reputation(ip):
            return True
    return False

# Function to check IP reputation using AbuseIPDB
async def check_ip_reputation(ip):
    url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&key={config['abuseipdb_api_key']}"
    async with ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            if data['data']['abuseConfidenceScore'] > 50:
                return True
    return False

# Main function to continuously monitor and protect the system
async def main():
    config = load_configuration()

    # Schedule periodic tasks
    schedule.every(1).minutes.do(monitor_office_processes)

    with MouseListener(on_click=on_click) as listener:
        while True:
            # Continuously monitor running processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    pid = proc.info['pid']
                    program_path = get_program_path(pid)
                    if is_office_document(program_path):
                        extract_macros(program_path)
                    if is_malicious_program(program_path, pid):
                        logging.warning(f"Malware detected: {program_path}")
                        terminate_process(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logging.error(f"Failed to monitor process {pid}: {e}")

            # Periodically check for malicious IPs and file reputations
            aioschedule.every(1).minutes.do(check_threat_intelligence)

async def on_click(x, y, button, pressed):
    active_window = get_active_window_title()
    pid = get_pid_from_window_title(active_window)
    program_path = get_program_path(pid)
    if is_office_document(program_path):
        extract_macros(program_path)
    if is_malicious_program(program_path, pid):
        logging.warning(f"Malware detected: {program_path}")
        terminate_process(pid)

# Function to check for known malicious signatures
async def is_malicious_signature(program_path):
    file_hash = hashlib.md5(open(program_path, 'rb').read()).hexdigest()
    if await check_file_reputation(file_hash):
        logging.warning(f"Malicious file detected: {program_path}")
        return True

# Function to check file reputation using VirusTotal
async def check_file_reputation(file_hash):
    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {"x-apikey": config['virustotal_api_key']}
    async with ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            data = await response.json()
            if data['data']['attributes']['last_analysis_stats']['malicious'] > 0:
                return True
    return False

# Function to check for malicious IPs and file reputations periodically
async def check_threat_intelligence():
    known_malicious_ips = config.get('known_malicious_ips', [])
    for proc in psutil.process_iter(['pid', 'name']):
        if is_office_document(proc.info['name']):
            process = psutil.Process(proc.info['pid'])
            try:
                connections = process.connections()
                for conn in connections:
                    ip = conn.raddr.ip
                    if ip in known_malicious_ips or await check_ip_reputation(ip):
                        logging.warning(f"Suspicious IP connection detected: {ip} from program {proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

# Function to get the active window title
def get_active_window_title():
    try:
        return pygetwindow.getActiveWindow().title
    except Exception as e:
        logging.error(f"Failed to get active window title: {e}")

# Function to get the PID from a window title
def get_pid_from_window_title(title):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if title.lower() in proc.info['name'].lower():
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# Function to get the program path from a PID
def get_program_path(pid):
    try:
        process = psutil.Process(pid)
        return process.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logging.error(f"Failed to get program path for PID {pid}: {e}")

# Function to terminate a process
def terminate_process(pid):
    try:
        psutil.Process(pid).terminate()
        logging.info(f"Process terminated: {pid}")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logging.error(f"Failed to terminate process {pid}: {e}")

# Schedule the main function to run
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
