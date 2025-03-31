import os
import sys
import hashlib
import requests
import socket
from urllib.parse import urlparse
import smtplib
from email.message import EmailMessage
from py3270 import Emulator
import subprocess
import shutil
import psutil
import schedule
import threading
import time
import logging

# Configure logging
logging.basicConfig(filename='protection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
EMAIL_USER = 'your_email@example.com'
EMAIL_PASSWORD = 'your_password'
MALICIOUS_EMAILS = ['malicious1@example.com', 'malicious2@example.com']
KNOWN_MALICIOUS_URLS = [
    'http://example.com/malware',
    'http://malware.example.com'
]
ALLOWED_DOWNLOAD_SOURCES = [
    'https://official-source.com',
    'https://another-safe-source.com'
]

# Helper functions
def hash_file(file_path):
    """Compute the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def install_libraries():
    """Install necessary libraries."""
    required_libraries = ['requests', 'py3270', 'hashlib', 'subprocess', 'os', 'shutil', 'socket', 'urllib', 'psutil', 'schedule']
    
    for library in required_libraries:
        try:
            __import__(library)
        except ImportError:
            logging.info(f"Installing {library}...")
            subprocess.run(['pip', 'install', library])

def scan_email_attachment(email_message):
    """Scan email attachments for malicious content."""
    logging.info("Scanning email attachments...")
    
    for part in email_message.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        
        filename = part.get_filename()
        if not filename:
            continue

        temp_path = os.path.join('/tmp', filename)
        with open(temp_path, 'wb') as f:
            f.write(part.get_payload(decode=True))

        file_hash = hash_file(temp_path)
        # Check against known malicious hashes
        with open('malicious_hashes.txt', 'r') as f:
            known_hashes = f.read().splitlines()
        
        if file_hash in known_hashes:
            logging.warning(f"Malicious attachment detected: {filename}")
            os.remove(temp_path)
        else:
            logging.info(f"Attachment {filename} is clean.")

def verify_software_download(url):
    """Verify the integrity of a software download."""
    logging.info(f"Verifying software download from {url}...")
    
    parsed_url = urlparse(url)
    if parsed_url.netloc not in ALLOWED_DOWNLOAD_SOURCES:
        logging.warning(f"Download from unknown source: {url}")
        return False
    
    response = requests.get(url, stream=True)
    temp_path = os.path.join('/tmp', os.path.basename(parsed_url.path))
    
    with open(temp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)

    file_hash = hash_file(temp_path)
    # Check against known good hashes
    with open('known_good_hashes.txt', 'r') as f:
        known_hashes = f.read().splitlines()
    
    if file_hash not in known_hashes:
        logging.warning(f"Download from {url} has an unknown hash.")
        os.remove(temp_path)
        return False

    shutil.move(temp_path, '/opt/downloads')
    return True

def block_malicious_websites(url):
    """Block access to known malicious websites."""
    logging.info(f"Blocking access to: {url}")
    
    if url in KNOWN_MALICIOUS_URLS:
        logging.warning(f"Blocked access to: {url}")
        return False
    return True

def prevent_drive_by_download(url, user_agent):
    """Prevent drive-by downloads from websites."""
    logging.info(f"Preventing drive-by download from: {url}")
    
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)
    
    if 'Content-Disposition' in response.headers:
        filename = os.path.join('/tmp', response.headers['Content-Disposition'].split('filename=')[-1])
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        file_hash = hash_file(filename)
        # Check against known malicious hashes
        with open('malicious_hashes.txt', 'r') as f:
            known_hashes = f.read().splitlines()
        
        if file_hash in known_hashes:
            logging.warning(f"Drive-by download detected: {filename}")
            os.remove(filename)
        else:
            logging.info(f"File {filename} is clean.")
    return True

def secure_network_sharing():
    """Secure network sharing and peer-to-peer connections."""
    logging.info("Securing network sharing...")
    
    # Check for open shares
    net_share_output = subprocess.check_output(['net', 'share']).decode()
    
    if "Share name" in net_share_output:
        logging.warning("Open network shares detected. Securing...")
        os.system('net share /delete *')
    
    # Check P2P connections
    p2p_processes = [proc for proc in psutil.process_iter() if 'torrent' in proc.name().lower()]
    
    for proc in p2p_processes:
        logging.warning(f"Terminating P2P process: {proc.name()}")
        proc.terminate()

def detect_social_engineering():
    """Detect and mitigate social engineering attempts."""
    logging.info("Detecting social engineering...")
    
    # Scan emails for phishing
    with open('emails.txt', 'r') as f:
        emails = f.read().splitlines()
    
    for email in emails:
        if any(malicious in email for malicious in MALICIOUS_EMAILS):
            logging.warning(f"Phishing attempt detected: {email}")
            # Send a warning email
            msg = EmailMessage()
            msg.set_content("This email may be a phishing attempt.")
            msg['Subject'] = 'Phishing Alert'
            msg['From'] = EMAIL_USER
            msg['To'] = email

            with smtplib.SMTP_SSL('smtp.example.com', 465) as smtp:
                smtp.login(EMAIL_USER, EMAIL_PASSWORD)
                smtp.send_message(msg)

def scan_usb_devices():
    """Scan USB and external devices for malicious content."""
    logging.info("Scanning USB and external devices...")
    
    # List all mounted drives
    mounted_drives = os.listdir('/media')

    for drive in mounted_drives:
        drive_path = os.path.join('/media', drive)
        
        # Scan files in the drive
        for root, dirs, files in os.walk(drive_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = hash_file(file_path)
                
                with open('malicious_hashes.txt', 'r') as f:
                    known_hashes = f.read().splitlines()
                
                if file_hash in known_hashes:
                    logging.warning(f"Malicious file detected: {file_path}")
                    os.remove(file_path)

def keep_system_up_to_date():
    """Keep the system up-to-date with the latest security patches."""
    logging.info("Updating system...")
    
    # Update package lists and upgrade installed packages
    subprocess.run(['sudo', 'apt-get', 'update'])
    subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'])

# Main function to run all tasks concurrently
def main():
    install_libraries()

    def task_runner(task, interval):
        while True:
            try:
                task()
            except Exception as e:
                logging.error(f"Error in {task.__name__}: {e}")
            time.sleep(interval)

    # List of tasks and their intervals
    tasks = [
        (scan_email_attachment, 60),
        (verify_software_download, 120),
        (block_malicious_websites, 60),
        (prevent_drive_by_download, 60),
        (secure_network_sharing, 300),
        (detect_social_engineering, 300),
        (scan_usb_devices, 300),
        (keep_system_up_to_date, 86400)
    ]

    # Start each task in a separate thread
    threads = []
    for task, interval in tasks:
        t = threading.Thread(target=task_runner, args=(task, interval))
        t.daemon = True
        t.start()
        threads.append(t)

    # Keep the main thread running to keep all threads active
    while True:
        time.sleep(10)  # Check every 10 seconds for new tasks

if __name__ == "__main__":
    main()
