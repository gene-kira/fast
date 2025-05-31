import os
import logging
import time
import psutil
import requests
import hashlib
import random
import threading
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import platform

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define critical system files and configuration settings
config_file = "system_protection_config.ini"
threat_intelligence_file = "threat_intelligence.csv"
known_safe_hashes_file = "known_safe_hashes.txt"

def load_configuration():
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                config[key] = value
    else:
        logging.warning("Configuration file not found. Using default settings.")
        config["scrape_interval"] = 3600  # Check threat intelligence every hour
        config["monitor_interval"] = 60   # Monitor processes every minute
    return config

def terminate_process(process_name):
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == process_name:
            try:
                p = psutil.Process(proc.info["pid"])
                p.terminate()
                logging.info(f"Terminated process {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

def collect_behavioral_data():
    behavioral_data = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            connections = proc.connections()
            files = proc.open_files()
            mem_info = proc.memory_info()

            behavioral_data.append({
                "pid": proc.pid,
                "name": proc.name(),
                "username": proc.username(),
                "connections": [conn.status for conn in connections],
                "files": len(files),
                "mem_info": mem_info.rss
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logging.error(f"Error collecting behavioral data for process {proc.pid}: {e}")
    return behavioral_data

def is_suspicious(data):
    network_threshold = 10
    file_threshold = 50
    memory_threshold = 100 * 1024 * 1024

    if sum(1 for conn in data.get("connections", []) if conn == "ESTABLISHED") >= network_threshold:
        return True
    if data["files"] > file_threshold:
        return True
    if data["mem_info"] > memory_threshold:
        return True

    return False

def monitor_system():
    config = load_configuration()

    while True:
        behavioral_data = collect_behavioral_data()
        for data in behavioral_data:
            if is_suspicious(data):
                logging.warning(f"Suspicious process {data['name']} detected. Terminating.")
                terminate_process(data["name"])

        time.sleep(int(config["monitor_interval"]))

def scrape_threat_intelligence(initial_urls, interval):
    while True:
        for url in initial_urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    threats = soup.find_all("a", class_="threat")
                    with open(threat_intelligence_file, "a") as f:
                        for threat in threats:
                            f.write(f"{threat.text}\n")
            except Exception as e:
                logging.error(f"Error scraping threat intelligence from {url}: {e}")
        time.sleep(interval)

def start_threat_intelligence_scraping():
    initial_urls = ["https://example.com/threats", "https://another-source.com/threats"]
    scrape_interval = load_configuration()["scrape_interval"]
    threading.Thread(target=scrape_threat_intelligence, args=(initial_urls, scrape_interval)).start()

def main():
    start_threat_intelligence_scraping()
    monitor_system()

if __name__ == "__main__":
    main()



