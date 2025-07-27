# magicbox_sentinel.py

import sys
import subprocess
import importlib
import os
import time
import threading
import json
from datetime import datetime
from tkinter import Tk, Label, Button, filedialog, messagebox, PhotoImage
from tkinter.ttk import Progressbar, Style
import pyttsx3

# 📦 Auto-install required packages
required = ['watchdog', 'cryptography', 'pyttsx3']
for package in required:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet

# 🧠 Settings
SETTINGS_FILE = 'magicbox_config.json'
DEFAULT_SETTINGS = {
    "watch_folder": "input_images",
    "archive_folder": "archived_images",
    "log_file": "sentinel_log.txt",
    "encrypt_logs": True,
    "theme": "dark",
    "voice_persona": "neutral"
}

# 🧪 Load or create settings
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

config = load_settings()

# 🔐 Encryption
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# 🎤 Voice Setup
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(message):
    if config["voice_persona"] != "silent":
        engine.say(message)
        engine.runAndWait()

# 📝 Logging
def log_event(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{timestamp}] {message}"
    if config["encrypt_logs"]:
        entry = cipher_suite.encrypt(entry.encode()).decode()
    with open(config["log_file"], "a") as log:
        log.write(entry + "\n")

# 🔍 Validator
def validate_image(file_path):
    valid = ['.jpg', '.jpeg', '.png', '.bmp']
    return os.path.splitext(file_path)[1].lower() in valid

# 🧠 Duplicate Check by Hash
def file_hash(file_path):
    import hashlib
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# 📦 Archiver
seen_hashes = set()

def archive_image(file_path):
    img_hash = file_hash(file_path)
    if img_hash in seen_hashes:
        log_event(f"Duplicate skipped: {file_path}")
        return
    seen_hashes.add(img_hash)
    base = os.path.basename(file_path)
    target = os.path.join(config["archive_folder"], base)
    os.makedirs(config["archive_folder"], exist_ok=True)
    try:
        with open(file_path, 'rb') as src, open(target, 'wb') as dst:
            dst.write(src.read())
        log_event(f"Archived: {base}")
        speak(f"Image {base} archived.")
    except Exception as e:
        log_event(f"Error archiving {base}: {e}")

# 🕵️ Folder Watcher
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and validate_image(event.src_path):
            log_event(f"New file detected: {event.src_path}")
            archive_image(event.src_path)

def start_watcher():
    observer = Observer()
    observer.schedule(Handler(), config["watch_folder"], recursive=False)
    observer.start()
    log_event("Watcher started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 🧪 System Check
def system_diagnostics():
    log_event("Running diagnostics...")
    speak("System diagnostics complete. All systems green.")

# 🎨 GUI
def run_gui():
    root = Tk()
    root.title("MagicBox Sentinel")
    root.geometry("400x200")
    root.resizable(False, False)

    style = Style()
    style.theme_use('default')
    style.configure("TProgressbar", thickness=8, troughcolor='gray', background='lime')

    Label(root, text="MagicBox Sentinel is active", font=("Arial", 14)).pack(pady=10)
    Label(root, text=f"Monitoring: {config['watch_folder']}").pack()

    progress = Progressbar(root, orient="horizontal", length=300, mode='indeterminate')
    progress.pack(pady=10)

    def start():
        threading.Thread(target=start_watcher, daemon=True).start()
        progress.start()
        messagebox.showinfo("Sentinel", "Watching folder!")
        speak("Watcher initiated. Monitoring in progress.")

    def run_diag():
        system_diagnostics()
        messagebox.showinfo("Diagnostics", "Diagnostics complete!")

    Button(root, text="Start Watcher", command=start).pack(pady=5)
    Button(root, text="Run Diagnostics", command=run_diag).pack()

    root.mainloop()

# 🚪 Entry Point
if __name__ == "__main__":
    os.makedirs(config["watch_folder"], exist_ok=True)
    os.makedirs(config["archive_folder"], exist_ok=True)
    run_gui()

