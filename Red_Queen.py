import os
import logging
import platform
import time
import csv
import psutil
import requests
from bs4 import BeautifulSoup
import hashlib
import random
from sklearn.ensemble import RandomForestClassifier

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

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound.playsound("output.mp3")
    os.remove("output.mp3")

def terminate_process(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            try:
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

def initiate_lockdown():
    speak("Initiating system lockdown. All non-essential processes will be terminated.")
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if not any([proc.info['name'] == name for name in ['python', 'cmd', 'explorer']]):
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def scan_and_remove_viruses():
    speak("Initiating virus scan.")
    result = subprocess.run(['malwarebytes', '--scan'], capture_output=True, text=True)
    if "Threats found:" in result.stdout:
        speak(f"Found {result.stdout.split('Threats found:')[1].split('\n')[0]} threats. Initiating removal process.")
        remove_result = subprocess.run(['malwarebytes', '--remove'], capture_output=True, text=True)
        if "Threats removed:" in remove_result.stdout:
            speak(f"Removed {remove_result.stdout.split('Threats removed:')[1].split('\n')[0]} threats.")
    else:
        speak("No viruses detected.")

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

def is_suspicious(data):
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

    return known_hashes

def monitor_system():
    config = load_configuration()
    os = get_os()

    known_safe_hashes = update_known_safe_hashes()
    
    while True:
        behavioral_data = collect_behavioral_data()
        
        for data in behavioral_data:
            if is_suspicious(data) and data['name'] not in known_safe_hashes:
                speak(f"Suspected process {data['name']} (PID: {data['pid']}) detected. Terminating.")
                terminate_process(data['name'])
        
        # Introduce randomness to simulate free will
        if random.random() < 0.1:
            attitude = random.choice(['casual', 'serious', 'playful'])
            if attitude == 'casual':
                speak("Just checking in, everything looks good so far.")
            elif attitude == 'serious':
                speak("I'm monitoring the system closely for any signs of threats.")
            else:
                speak("Hey there! I noticed something interesting. Let's keep an eye on things.")

        time.sleep(int(config['monitor_interval']))

def main():
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    known_safe_hashes = update_known_safe_hashes()

    # Scrape threat intelligence
    initial_urls = ['https://example.com/threats', 'https://another-source.com/threats']
    scrape_interval = int(config['scrape_interval'])

    def start_scraping():
        threading.Thread(target=scrape_threat_intelligence, args=(initial_urls, scrape_interval)).start()

    # Start monitoring the system
    monitor_thread = threading.Thread(target=monitor_system)
    monitor_thread.start()

    start_scraping()

if __name__ == "__main__":
    main()
