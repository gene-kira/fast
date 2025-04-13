import psutil
import os
import subprocess
from gtts import gTTS
from playsound import playsound
import threading
import logging
import time
import json
import requests

# Ensure all necessary libraries are installed
try:
    from tqdm import tqdm
except ImportError:
    print("Installing required libraries...")
    subprocess.check_call(["pip", "install", "tqdm"])

def speak(text):
    """Speak the given text using text-to-speech."""
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")

def load_configuration():
    with open('config.json', 'r') as f:
        return json.load(f)

def get_os():
    """Determine the operating system."""
    if os.name == 'nt':
        return 'Windows'
    else:
        return 'Unix'

def get_os_specific_commands(os):
    """Return OS-specific commands for antivirus scanning."""
    if os == 'Windows':
        return {
            'scan': ['powershell', '-Command', 'Start-MpScan'],
            'remove': ['powershell', '-Command', 'Remove-MpThreat']
        }
    else:
        return {
            'scan': ['clamscan', '--infected', '--recursive', '/'],
            'remove': ['clamscan', '--infected', '--recursive', '--remove', '/']
        }

def collect_behavioral_data():
    """Collect detailed behavioral data from running processes."""
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
    """Determine if a process is suspicious."""
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

def terminate_process(process_name):
    """Terminate a specific process by name."""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            try:
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

def initiate_lockdown():
    """Initiate system lockdown by terminating all non-essential processes."""
    essential_processes = ['python', 'cmd', 'explorer']
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if not any(proc.info['name'] == name for name in essential_processes):
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                speak(f"Terminated process {proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def scan_and_remove_viruses():
    """Initiate virus scan using the system's built-in antivirus tool."""
    config = load_configuration()
    os = get_os()
    commands = get_os_specific_commands(os)

    speak("Initiating virus scan.")
    result = subprocess.run(commands['scan'], capture_output=True, text=True)
    if "Threats found:" in result.stdout:
        threats_found = int(result.stdout.split('Threats found:')[1].split('\n')[0])
        speak(f"Found {threats_found} threats. Initiating removal process.")
        remove_result = subprocess.run(commands['remove'], capture_output=True, text=True)
        if "Threats removed:" in remove_result.stdout:
            threats_removed = int(remove_result.stdout.split('Threats removed:')[1].split('\n')[0])
            speak(f"Removed {threats_removed} threats.")
    else:
        speak("No viruses detected.")

def handle_command(command):
    """Handle user commands with a human-like attitude and a sense of humor."""
    if command == 'lockdown':
        speak("Alright, let's get this place locked down. It's like putting away the dishes before a party starts. Starting lockdown now.")
        initiate_lockdown()
    elif command.startswith('terminate'):
        process_name = command.split(' ')[1]
        speak(f"Time to bid farewell to {process_name}. Goodbye, old friend!")
        terminate_process(process_name)
    elif command == 'scan':
        speak("Time to give this system a thorough check. Let's see if we can catch any little bugs hiding in the corners.")
        scan_and_remove_viruses()
    else:
        speak("Unknown command. Valid commands are: lockdown, terminate [process], and scan. It’s like learning a new game; you’ll get the hang of it soon.")

def main():
    """Main loop to interact with the Red Queen."""
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
    """Continuously monitor and report on system health."""
    speak("I'm keeping a watchful eye over the system. You can rest easy knowing I've got your back.")
    while True:
        processes = collect_behavioral_data()
        for process in processes:
            if is_suspicious(process):
                speak(f"Hey, looks like we have a suspicious process running: {process['name']} by user {process['username']}. Might want to take a closer look at this one.")
        time.sleep(60)

if __name__ == "__main__":
    main()
