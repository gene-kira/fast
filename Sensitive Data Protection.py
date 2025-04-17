import os
import subprocess
import json
import hashlib
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define log file
LOGFILE = "/var/log/system_protection.log"

# Function to log messages
def log(message):
    with open(LOGFILE, 'a') as logfile:
        logfile.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Function to monitor and inspect kernel modules
def monitor_kernel_modules():
    try:
        # Get the list of currently loaded modules
        current_modules = subprocess.check_output("lsmod", shell=True).decode().strip().split('\n')
        
        # Load the previous state from a file or create an initial state
        if os.path.exists('/var/state/previous_modules.json'):
            with open('/var/state/previous_modules.json', 'r') as f:
                previous_modules = json.load(f)
        else:
            with open('/var/state/previous_modules.json', 'w') as f:
                json.dump(current_modules, f)
            log("Initial state of kernel modules recorded.")
            return
        
        # Compare current and previous states
        if set(current_modules) != set(previous_modules):
            log("Detected changes in kernel modules. Inspecting...")
            
            # Calculate hash for current and previous module lists
            current_hash = hashlib.sha256('\n'.join(current_modules).encode()).hexdigest()
            previous_hash = hashlib.sha256('\n'.join(previous_modules).encode()).hexdigest()
            
            if current_hash != previous_hash:
                log("Kernel modules have been modified. Checking for rogue modules.")
                
                # Identify new modules
                new_modules = set(current_modules) - set(previous_modules)
                for module in new_modules:
                    log(f"New kernel module detected: {module}")
                    # You can add additional checks here, such as verifying the module's integrity or checking against a whitelist

            # Update the state file with the current modules
            with open('/var/state/previous_modules.json', 'w') as f:
                json.dump(current_modules, f)
    except Exception as e:
        log(f"Error monitoring kernel modules: {e}")

# Function to secure storage of sensitive data
def secure_data_storage(data):
    try:
        # Define the path for storing encrypted data
        data_file = '/var/data/secure_data.json'
        
        # Encrypt the data using a simple XOR cipher (for demonstration purposes)
        key = os.urandom(len(str(data).encode()))
        encrypted_data = bytes([b ^ k for b, k in zip(str(data).encode(), key)])
        
        # Store the encrypted data and the key
        with open(data_file, 'wb') as f:
            f.write(encrypted_data + b'\n' + key)
            log("Sensitive data stored securely.")
    except Exception as e:
        log(f"Error securing data: {e}")

# Function to detect and block rogue software
class RogueSoftwareDetector(FileSystemEventHandler):
    def __init__(self, path_to_watch):
        self.path_to_watch = path_to_watch
        self.known_files = set(os.listdir(path_to_watch))
    
    def on_modified(self, event):
        if not event.is_directory:
            file_path = os.path.join(self.path_to_watch, os.path.basename(event.src_path))
            if file_path not in self.known_files:
                log(f"New file detected: {file_path}")
                self.inspect_file(file_path)
    
    def inspect_file(self, file_path):
        try:
            # Check file signature
            with open(file_path, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
            
            # Load known good hashes from a file or create an initial state
            if os.path.exists('/var/state/known_good_hashes.json'):
                with open('/var/state/known_good_hashes.json', 'r') as f:
                    known_good_hashes = json.load(f)
            else:
                with open('/var/state/known_good_hashes.json', 'w') as f:
                    json.dump([], f)
            
            # Check if the file hash is in the list of known good hashes
            if file_hash not in known_good_hashes:
                log(f"File {file_path} has an unknown hash. Performing additional checks.")
                
                # Perform additional checks, such as signature verification or heuristic analysis
                # For now, we will just quarantine the file
                quarantine_path = os.path.join('/var/quarantine', os.path.basename(file_path))
                os.rename(file_path, quarantine_path)
                log(f"File {file_path} quarantined to {quarantine_path}")
        except Exception as e:
            log(f"Error inspecting file: {e}")

def main():
    # Ensure necessary directories exist
    for dir in ['/var/log', '/var/state', '/var/data', '/var/quarantine']:
        os.makedirs(dir, exist_ok=True)
    
    # Monitor and inspect kernel modules
    monitor_kernel_modules()
    
    # Secure storage of sensitive data
    secure_data = {
        "cookies": ["cookie1", "cookie2"],
        "passwords": ["password1", "password2"],
        "personal_info": {"name": "John Doe", "email": "john.doe@example.com"}
    }
    secure_data_storage(secure_data)
    
    # Set up file system monitoring for rogue software detection
    path_to_watch = '/usr/local/bin'  # Adjust this to the directory you want to monitor
    event_handler = RogueSoftwareDetector(path_to_watch)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()
    
    log("System protection script started.")
    
    try:
        while True:
            time.sleep(10)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
        log("System protection script stopped.")
    finally:
        observer.join()

if __name__ == "__main__":
    main()
