import os
import subprocess
import logging
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
from urllib.parse import urlparse
import psutil
import socket
import hashlib

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Auto-load libraries
def auto_load_libraries():
    required_libraries = ['os', 'subprocess', 'logging', 'requests', 'json', 'sklearn.ensemble', 'sklearn.feature_extraction.text', 'sklearn.decomposition', 'sklearn.model_selection', 'sklearn.metrics', 'numpy', 'urllib.parse', 'psutil', 'socket', 'hashlib']
    for lib in required_libraries:
        try:
            __import__(lib)
            logging.info(f"Loaded library: {lib}")
        except ImportError as e:
            logging.error(f"Failed to load library: {lib} - {e}")

# Machine Learning Model Setup
def train_model(data):
    # Split data into features and labels
    X = [urlparse(url).netloc for url in data['url']]
    y = data['label']

    # Vectorize the URLs
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_vectorized, y)
    
    best_model = grid_search.best_estimator_
    
    return best_model, vectorizer

# Function to predict if a URL is malicious
def predict_malicious_url(url, model, vectorizer):
    url_vectorized = vectorizer.transform([urlparse(url).netloc])
    prediction = model.predict(url_vectorized)
    return prediction[0]

# Network Traffic Monitoring and Blocking
def monitor_network_traffic(model, vectorizer):
    urls = [
        'http://example.com',
        'http://malicious-site.com',
        'http://trusted-site.com'
    ]
    
    for url in urls:
        if predict_malicious_url(url, model, vectorizer) == 1:
            logging.warning(f"Blocking malicious URL: {url}")
            # Implement blocking logic here

# Terminate a network connection by port
def terminate_connection(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                p = psutil.Process(conn.pid)
                logging.warning(f"Terminating process {p.name()} using port {port}")
                p.terminate()
                p.wait(timeout=3)  # Wait for the process to terminate
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.error(f"Failed to terminate connection on port {port}: {e}")

# Detect P2P connections on all open ports
def detect_p2p_connections(open_ports):
    p2p_ports = [6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889, 6890]  # Common BitTorrent ports
    for port in open_ports:
        if port in p2p_ports:
            logging.warning(f"Detected P2P connection on port {port}")
            terminate_connection(port)

# Block BitTorrent programs
def block_bittorrent_programs():
    bittorrent_processes = ['transmission', 'utorrent', 'deluge', 'qBittorrent']
    for process in psutil.process_iter(['pid', 'name']):
        if any(bt in process.info['name'].lower() for bt in bittorrent_processes):
            logging.warning(f"Detected and terminating BitTorrent program: {process.info['name']}")
            terminate_process(process)

# Monitor File System for Encryption Attempts
def monitor_file_system():
    while True:
        for process in psutil.process_iter(['pid', 'name']):
            try:
                if is_malicious_process(process):
                    logging.warning(f"Detected rogue program trying to encrypt files: {process.info['name']}")
                    terminate_process(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.error(f"Failed to monitor file system: {e}")

# Check if a process is malicious using behavior analysis and signature matching
def is_malicious_process(process):
    # Example heuristic: check for known malicious filenames or behaviors
    if 'malicious' in process.info['name'].lower():
        return True

    # Behavior Analysis
    if any(is_suspicious_behavior(conn) for conn in psutil.net_connections() if conn.pid == process.info['pid']):
        return True

    # Signature Matching
    if is_malicious_signature(process):
        return True

    return False

# Check for suspicious behavior patterns
def is_suspicious_behavior(connection):
    # Example: check for connections to known malicious IP addresses or ports
    if connection.raddr.ip in ['192.168.1.100', '10.0.0.1']:
        return True
    if connection.raddr.port in [6881, 6882, 6883]:
        return True
    return False

# Check for known malicious signatures
def is_malicious_signature(process):
    # Example: check the process's hash against a database of known malware hashes
    process_path = psutil.Process(process.info['pid']).exe()
    process_hash = hashlib.md5(open(process_path, 'rb').read()).hexdigest()
    if process_hash in load_malware_database():
        return True
    return False

# Load a database of known malware hashes
def load_malware_database():
    # Example: read from a file or an API
    with open('malware_hashes.txt', 'r') as f:
        return set(f.read().splitlines())

# Terminate a process
def terminate_process(process):
    try:
        p = psutil.Process(process.info['pid'])
        p.terminate()
        p.wait(timeout=3)  # Wait for the process to terminate
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logging.error(f"Failed to terminate process: {e}")

# Main function
def main():
    auto_load_libraries()
    
    data = {
        'url': ['http://example.com', 'http://malicious-site.com', 'http://trusted-site.com'],
        'label': [0, 1, 0]
    }
    
    model, vectorizer = train_model(data)
    
    pca = PCA(n_components=2)  # Example: reduce to 2 components
    unique_ports_data = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]  # Example data for known ports
    unique_ports = [123, 456]  # Example known ports
    pca.fit(unique_ports_data)
    
    target_ip = '127.0.0.1'  # Example target IP
    open_ports = scan_ports(target_ip)
    detect_and_terminate_anomalies(open_ports, model, unique_ports, pca)
    
    monitor_network_traffic(model, vectorizer)
    block_bittorrent_programs()
    
    # Start monitoring file system in a separate thread
    import threading
    file_system_monitor_thread = threading.Thread(target=monitor_file_system)
    file_system_monitor_thread.daemon = True  # Daemonize the thread to run in background
    file_system_monitor_thread.start()

# Detect and terminate anomalies using machine learning models
def detect_and_terminate_anomalies(open_ports, model, unique_ports, pca):
    for port in open_ports:
        if port not in unique_ports:
            logging.warning(f"Detected anomaly on port {port}")
            terminate_connection(port)

# Scan for open ports
def scan_ports(target_ip, port_range=(1, 65535)):
    open_ports = []
    for port in range(*port_range):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((target_ip, port))
            if result == 0:
                open_ports.append(port)
    return open_ports

if __name__ == "__main__":
    main()
