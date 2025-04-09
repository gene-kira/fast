import os
import logging
import platform
import time
import csv
import psutil
from sklearn.ensemble import RandomForestClassifier
import requests
from bs4 import BeautifulSoup
import hashlib
import sys

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define configuration file path
config_file = 'system_protection_config.ini'

def get_os():
    return platform.system()

def load_configuration():
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                config[key] = value
    else:
        logging.warning("Configuration file not found. Using default settings.")
        config['scrape_interval'] = 3600  # Check threat intelligence every hour
        config['monitor_interval'] = 60   # Monitor processes every minute
        config['threat_intelligence_file'] = 'threat_intelligence.csv'
        config['known_safe_hashes_file'] = 'known_safe_hashes.txt'
    return config

def get_os_specific_commands(os):
    if os == 'Windows':
        return {
            'psutil': psutil,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'hashlib': hashlib
        }
    elif os in ['Linux', 'Darwin']:
        return {
            'psutil': psutil,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'hashlib': hashlib
        }

# Main function to initialize and start monitoring
def main():
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    # Define paths for threat intelligence and known safe hashes
    threat_intelligence_file = config['threat_intelligence_file']
    known_safe_hashes_file = config['known_safe_hashes_file']

    # Load the machine learning model
    global model
    try:
        model = load_model()
    except Exception as e:
        logging.error(f"Error loading machine learning model: {e}")

    # Scrape threat intelligence from multiple sources
    initial_urls = ['https://example.com/threats', 'https://threatintelligence.net']
    scrape_threat_intelligence(initial_urls, config['scrape_interval'])

    # Train the machine learning model using threat intelligence data
    train_model(threat_intelligence_file)

    # Update the list of known safe hashes
    update_known_safe_hashes()

    # Start monitoring processes and files
    monitor_processes(config['monitor_interval'])

def load_model():
    try:
        from keras.models import load_model
        return load_model('ai_threat_detection_model.h5')
    except Exception as e:
        logging.error(f"Error loading machine learning model: {e}")

def train_model(threat_intelligence_file):
    # Collect behavioral data
    behavioral_data = collect_behavioral_data()

    # Convert the collected data into a format suitable for machine learning
    X = []
    y = []

    for data in behavioral_data:
        features = [
            int('python' in data['name'] or 'java' in data['name']),
            any('ai' in arg.lower() for arg in data['cmdline']),
            sum(1 for conn in data.get('connections', []) if conn.status == psutil.CONN_ESTABLISHED and conn.type == psutil.SOCK_STREAM) > 0,
            len(data['files']) > 10,
            data['mem_info'] > 100 * 1024 * 1024
        ]
        X.append(features)
        y.append(1 if is_suspicious_content(data) else 0)

    # Train the machine learning model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X, y)
    except Exception as e:
        logging.error(f"Error training machine learning model: {e}")

def collect_behavioral_data():
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

def is_suspicious_content(data):
    # Define thresholds and conditions for suspicion
    ai_name_threshold = 0.5
    cmd_arg_threshold = 0.5
    network_threshold = 1
    file_threshold = 10
    memory_threshold = 100 * 1024 * 1024

    # Check AI-specific names
    if 'python' in data['name'].lower() or 'java' in data['name'].lower():
        return True

    # Check command line arguments for AI-specific keywords
    if any('ai' in arg.lower() for arg in data['cmdline']):
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

def scrape_threat_intelligence(initial_urls, interval):
    while True:
        for url in initial_urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    threats = soup.find_all('a', class_='threat')
                    with open(threat_intelligence_file, 'a') as f:
                        for threat in threats:
                            f.write(f"{threat.text}\n")
            except Exception as e:
                logging.error(f"Error scraping threat intelligence from {url}: {e}")
        time.sleep(interval)

def update_known_safe_hashes():
    known_hashes = set()
    if os.path.exists(known_safe_hashes_file):
        with open(known_safe_hashes_file, 'r') as f:
            for line in f:
                known_hashes.add(line.strip())

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

    for file in critical_files:
        try:
            with open(file, 'rb') as f:
                content = f.read()
                file_hash = hashlib.md5(content).hexdigest()
                known_hashes.add(file_hash)
        except Exception as e:
            logging.error(f"Error calculating hash for {file}: {e}")

    with open(known_safe_hashes_file, 'w') as f:
        for hash in known_hashes:
            f.write(f"{hash}\n")

def monitor_processes(interval):
    while True:
        try:
            processes = psutil.process_iter(['pid', 'name', 'username'])
            for proc in processes:
                try:
                    behavioral_data = {
                        'pid': proc.pid,
                        'name': proc.name(),
                        'username': proc.username(),
                        'cmdline': proc.cmdline(),
                        'connections': [conn.status for conn in proc.connections()],
                        'files': len(proc.open_files()),
                        'mem_info': proc.memory_info().rss
                    }

                    if is_suspicious_content(behavioral_data):
                        logging.warning(f"Suspicious process detected: {behavioral_data}")

                    prediction = model.predict([[
                        int('python' in behavioral_data['name'] or 'java' in behavioral_data['name']),
                        any('ai' in arg.lower() for arg in behavioral_data['cmdline']),
                        sum(1 for conn in behavioral_data.get('connections', []) if conn.status == psutil.CONN_ESTABLISHED and conn.type == psutil.SOCK_STREAM) > 0,
                        len(behavioral_data['files']) > 10,
                        behavioral_data['mem_info'] > 100 * 1024 * 1024
                    ])

                    if prediction[0] == 1:
                        logging.warning(f"Machine learning model detected suspicious process: {behavioral_data}")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logging.error(f"Error monitoring process {proc.pid}: {e}")
        except Exception as e:
            logging.error(f"Error in main monitoring loop: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    main()
