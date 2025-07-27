import time, threading, pyttsx3
from datetime import datetime
from utils import get_network_context, is_usb, is_local_network, purge_file

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def run_watchdog():
    print("🛡️ Sentinel online...")
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    speak(f"{greet}. System Data Guard online.")

    # Simulated file paths
    test_paths = [
        "C:\\Users\\User\\Documents\\safe.txt",
        "USB_Device\\music.zip",
        "D:\\Vault\\archive.pdf",
        "\\\\Server\\share\\sensitive.docx"
    ]

    for path in test_paths:
        watchdog(path)
        time.sleep(5)

def watchdog(path):
    context = get_network_context(path)
    if is_usb(context) or is_local_network(context):
        print(f"✅ Safe transfer: {path}")
    else:
        print(f"⚠️ External transfer detected: {path}")
        speak("Warning. External transfer detected. File will self-destruct in 30 seconds.")
        threading.Timer(30, lambda: purge_file(path)).start()

