import psutil
import json
import hashlib
import subprocess
from sklearn.ensemble import RandomForestClassifier
import threading
import sys

# Global variables for data storage and configuration
data_storage = []
model = None

def collect_behavioral_data():
    """Collect behavioral data from system processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'ppid']):
        try:
            processes.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'username': proc.info['username'],
                'ppid': proc.info['ppid']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def collect_network_data():
    """Collect network data for analysis."""
    connections = []
    for conn in psutil.net_connections(kind='inet'):
        connections.append({
            'laddr': conn.laddr,
            'raddr': conn.raddr,
            'status': conn.status,
            'pid': conn.pid
        })
    return connections

def collect_file_access_data():
    """Collect file access data for analysis."""
    files = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for handle in proc.open_files():
                files.append({
                    'pid': proc.info['pid'],
                    'file': handle.path
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return files

def collect_user_commands():
    """Collect user commands and interactions."""
    commands = []
    while True:
        command = input("Enter command: ").strip().lower()
        commands.append(command)
        handle_command(command)
        if len(commands) % 10 == 0:
            save_commands_to_file(commands)

def train_model(data):
    """Train a model to predict suspicious processes."""
    features = [d['pid'], d['name'], d['username'], d['ppid'] for d in data]
    labels = [1 if is_suspicious(d) else 0 for d in data]  # 1 for suspicious, 0 for not
    model = RandomForestClassifier()
    model.fit(features, labels)
    return model

def predict_suspicious(model, process):
    """Predict if a process is suspicious using the trained model."""
    features = [process['pid'], process['name'], process['username'], process['ppid']]
    prediction = model.predict([features])
    return bool(prediction[0])

def generate_optimized_code():
    """Generate optimized code for the current environment."""
    # Analyze system resource usage
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent

    if cpu_usage > 70 or memory_usage > 80:
        speak("System resources are high. Optimizing code to reduce load.")
        optimize_code_for_resources()
    else:
        speak("System resources are low. No optimization needed.")

def optimize_code_for_resources():
    """Optimize code for resource efficiency."""
    # Optimize CPU usage
    for function in [collect_behavioral_data, collect_network_data, collect_file_access_data]:
        if function.__name__ == 'collect_behavioral_data':
            interval = 60  # Increase collection interval to reduce CPU load
        elif function.__name__ == 'collect_network_data':
            interval = 300  # Collect network data less frequently
        else:
            interval = 120  # Collect file access data less frequently

    for func in [monitor_system, handle_command]:
        if func.__name__ == 'monitor_system':
            sleep_time = 60  # Increase sleep time to reduce CPU load
        elif func.__name__ == 'handle_command':
            sleep_time = 5  # Decrease sleep time to improve responsiveness

    # Optimize memory usage
    global data_storage
    if len(data_storage) > 10000:
        data_storage = data_storage[-5000:]  # Keep only the last 5000 entries

def load_configuration():
    """Load configuration and verify integrity."""
    with open('config.json', 'r') as file:
        config = json.load(file)
    if not verify_integrity(config):
        speak("Configuration file has been tampered with. Restoring default configuration.")
        config = {
            "os": get_os(),
            "commands": get_os_specific_commands(get_os())
        }
    return config

def verify_integrity(data):
    """Verify the integrity of the data using a hash."""
    original_hash = hashlib.sha256(json.dumps(data).encode()).hexdigest()
    with open('config.json', 'r') as file:
        stored_data = json.load(file)
    stored_hash = hashlib.sha256(json.dumps(stored_data).encode()).hexdigest()
    return original_hash == stored_hash

def get_os():
    """Determine the operating system."""
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform.startswith('linux'):
        return 'linux'
    else:
        return 'unknown'

def get_os_specific_commands(os):
    """Get OS-specific commands for scanning and removal."""
    if os == 'windows':
        return {
            "scan": ["powershell", "-Command", "Get-MpComputerStatus"],
            "remove": ["powershell", "-Command", "Remove-MpThreat"]
        }
    elif os == 'linux':
        return {
            "scan": ["clamscan", "--infected", "--recursive", "/"],
            "remove": ["clamscan", "--infected", "--recursive", "--remove", "/"]
        }
    else:
        return {}

def is_suspicious(process):
    """Determine if a process is suspicious based on known patterns."""
    # Example criteria for suspicion
    if process['name'].startswith('mal'):
        return True
    if process['username'] == 'Unknown':
        return True
    if process['ppid'] == 0:
        return True
    return False

def main():
    """Main loop to interact with the AI and continuously monitor the system."""
    speak("Red Queen online. Awaiting your orders, dear user.")
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    # Start monitoring the system
    monitor_thread = threading.Thread(target=monitor_system)
    monitor_thread.start()

    while True:
        command = input("Enter command: ").strip().lower()
        handle_command(command)

def monitor_system():
    """Monitor the system for suspicious activities."""
    global data_storage, model
    if not model:
        model = train_model(collect_behavioral_data())

    while True:
        processes = collect_behavioral_data()
        for process in processes:
            if predict_suspicious(model, process):
                speak(f"Suspicious process detected: {process}")
        generate_optimized_code()
        time.sleep(60)  # Adjust sleep interval as needed

def handle_command(command):
    """Handle user commands."""
    if command == 'scan':
        result = subprocess.run(commands['scan'], capture_output=True)
        if 'infected' in result.stdout.decode().lower():
            speak("Malware detected. Initiating removal.")
            subprocess.run(commands['remove'])
        else:
            speak("No malware detected.")
    elif command == 'optimize':
        generate_optimized_code()
    # Add more commands as needed

if __name__ == "__main__":
    main()
