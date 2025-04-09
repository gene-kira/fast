import os
import subprocess
import time
import psutil
import cv2
import pyaudio

# Auto-install necessary libraries
def install_libraries():
    required_libraries = ['psutil', 'pyaudio', 'opencv-python']
    for library in required_libraries:
        try:
            __import__(library)
        except ImportError:
            print(f"Installing {library}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Initialize the script
def initialize():
    install_libraries()
    os.system("cls" if os.name == "nt" else "clear")
    print("Security Monitor Initialized")

# Function to check for unauthorized webcam access
def monitor_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unauthorized webcam access detected!")
            os.system("taskkill /F /IM capstone.exe")  # Example command to kill a suspicious process
            break
        time.sleep(1)  # Check every second
    cap.release()

# Function to check for unauthorized microphone access
def monitor_microphone():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True)
    while True:
        data = stream.read(1024)
        if len(data) == 0:
            print("Unauthorized microphone access detected!")
            os.system("taskkill /F /IM capstone.exe")  # Example command to kill a suspicious process
            break
        time.sleep(1)  # Check every second
    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to monitor active processes for known malicious behavior
def monitor_processes():
    suspicious_processes = ["malware.exe", "rogue.exe", "suspicious_program.exe"]
    while True:
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['name'] in suspicious_processes:
                print(f"Suspicious process detected: {process.info['name']}")
                os.system(f"taskkill /F /IM {process.info['name']}")  # Kill the suspicious process
        time.sleep(5)  # Check every 5 seconds

# Main function to run all monitoring tasks
def main():
    initialize()

    print("Starting security monitors...")
    
    # Start webcam monitoring in a separate thread
    import threading
    webcam_thread = threading.Thread(target=monitor_webcam)
    webcam_thread.daemon = True
    webcam_thread.start()
    
    # Start microphone monitoring in a separate thread
    mic_thread = threading.Thread(target=monitor_microphone)
    mic_thread.daemon = True
    mic_thread.start()
    
    # Start process monitoring in the main thread
    monitor_processes()

if __name__ == "__main__":
    import sys
    main()
