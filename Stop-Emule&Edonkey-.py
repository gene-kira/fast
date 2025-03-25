import os
import re
import time
from scapy.all import sniff, IP, UDP
import psutil
import subprocess
import sys

# List of required libraries
required_libraries = ['scapy', 'psutil']

def install_library(library):
    try:
        __import__(library)
    except ImportError:
        print(f"{library} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

def ensure_libraries_installed():
    for library in required_libraries:
        install_library(library)

def main_script():
    # Define the network ports to monitor
    EMULE_PORT = 4242  # eMule KAD default port
    EDonkey_PORT = 4662  # eDonkey P2P default port

    def packet_callback(packet):
        if IP in packet:
            ip_layer = packet[IP]
            if UDP in packet and (ip_layer.dport == EMULE_PORT or ip_layer.dport == EDonkey_PORT):
                print(f"Detected traffic on port {ip_layer.dport} from {ip_layer.src}")
                process_id = find_process_by_port(ip_layer.dport)
                if process_id:
                    terminate_and_remove_program(process_id)

    def find_process_by_port(port):
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                conns = proc.connections()
                for conn in conns:
                    if conn.laddr.port == port:
                        return proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def terminate_and_remove_program(process_id):
        try:
            process = psutil.Process(process_id)
            process.terminate()
            process.wait()  # Ensure the process is terminated
            print(f"Terminated process {process.name()} with PID {process.pid}")
            remove_program_files(process.name())
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Failed to terminate process: {e}")

    def remove_program_files(program_name):
        program_paths = {
            "aMule": ["/usr/bin/aMule", "~/.aMule"],
            "emule": ["C:\\Program Files\\eMule", "%APPDATA%\\eMule"],
            "edonkey": ["C:\\Program Files\\Edonkey", "%APPDATA%\\Edonkey"]
        }
        for path in program_paths.get(program_name.lower(), []):
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                if os.path.isdir(expanded_path):
                    os.system(f'rm -rf "{expanded_path}"')
                    print(f"Removed directory: {expanded_path}")
                elif os.path.isfile(expanded_path):
                    os.remove(expanded_path)
                    print(f"Removed file: {expanded_path}")

    # Start the network monitoring
    sniff(filter=f"udp port {EMULE_PORT} or udp port {EDonkey_PORT}", prn=packet_callback, store=0)

if __name__ == "__main__":
    ensure_libraries_installed()
    main_script()
