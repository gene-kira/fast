import os
import psutil
import hashlib
from threading import Thread
import time

# Auto-load required libraries
try:
    import requests
except ImportError:
    os.system("pip install requests")

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    os.system("pip install watchdog")

try:
    import pynvml
except ImportError:
    os.system("pip install nvidia-ml-py3")

# Define the main protection script

def monitor_cpu_usage():
    """Monitor CPU usage and detect unusual activity"""
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:  # Threshold for high CPU usage
            print(f"High CPU Usage detected: {cpu_percent}%")
            take_action("CPU", "High Usage")

def monitor_network_activity():
    """Monitor network activity for unusual connections"""
    while True:
        connections = psutil.net_connections()
        for conn in connections:
            if not conn.laddr or not conn.raddr:
                continue
            if is_suspicious(conn):
                print(f"Suspicious Network Connection: {conn}")
                take_action("Network", "Suspicious Connection")

def is_suspicious(connection):
    """Check if a network connection is suspicious"""
    # Define your own criteria for what constitutes a suspicious connection
    return (connection.status == psutil.CONN_ESTABLISHED and
            not any(ip in str(connection.raddr) for ip in ['127.0.0.1', 'localhost']))

def monitor_file_system():
    """Monitor file system changes"""
    class FileChangeHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if is_critical_file(event.src_path):
                print(f"Critical file modified: {event.src_path}")
                take_action("File", "Modified")

    observer = Observer()
    observer.schedule(FileChangeHandler(), path='/', recursive=True)
    observer.start()

def is_critical_file(file_path):
    """Check if the modified file is critical"""
    # Define your own list of critical files
    critical_files = [
        '/etc/passwd',
        '/etc/shadow',
        '/etc/sudoers',
        '/var/log/auth.log'
    ]
    return file_path in critical_files

def monitor_memory():
    """Monitor memory for known backdoor signatures"""
    while True:
        processes = psutil.process_iter(['pid', 'name'])
        for proc in processes:
            try:
                mem_info = proc.memory_info()
                if is_suspicious_process(proc, mem_info):
                    print(f"Suspicious Process: {proc.info}")
                    take_action("Memory", "Suspicious Process")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

def is_suspicious_process(process, memory_info):
    """Check if the process has a known backdoor signature"""
    # Define your own list of suspicious processes and memory usage
    suspicious_processes = ['malicious.exe', 'backdoor.bin']
    return (process.name() in suspicious_processes or
            memory_info.rss > 100 * 1024 * 1024)  # More than 100MB

def take_action(component, issue):
    """Take action based on the detected threat"""
    print(f"Taking action for {component}: {issue}")
    if component == "CPU":
        os.system("echo 'High CPU Usage Detected' | wall")
    elif component == "Network":
        os.system(f"iptables -A INPUT -s {str(issue)} -j DROP")
    elif component == "File":
        os.system(f"chattr +i {issue}")
    elif component == "Memory":
        os.system(f"kill -9 {process.pid}")

def main():
    # Start monitoring threads
    cpu_thread = Thread(target=monitor_cpu_usage)
    network_thread = Thread(target=monitor_network_activity)
    file_system_thread = Thread(target=monitor_file_system)
    memory_thread = Thread(target=monitor_memory)

    cpu_thread.start()
    network_thread.start()
    file_system_thread.start()
    memory_thread.start()

if __name__ == "__main__":
    main()
