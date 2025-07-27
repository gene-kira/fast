# sentinel_core.py

import sys
import subprocess
import importlib
import os
import time
import threading
from datetime import datetime
from tkinter import Tk, Label, Button, filedialog, messagebox

# 📦 Library Autoloader
required_packages = ['watchdog', 'cryptography']
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet

# 📁 Setup Paths
WATCH_FOLDER = "input_images"
ARCHIVE_FOLDER = "archived_images"
LOG_FILE = "sentinel_log.txt"

# 🔐 Generate encryption key (stored temporarily; later to be persisted securely)
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# 📜 Logging Utility
def log_event(message, encrypt=False):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{timestamp}] {message}"
    if encrypt:
        entry = cipher_suite.encrypt(entry.encode()).decode()
    with open(LOG_FILE, "a") as log:
        log.write(entry + "\n")

# 🔍 Image Validator Stub (expandable)
def validate_image(file_path):
    valid_extensions = ['.jpg', '.png', '.jpeg']
    return os.path.splitext(file_path)[1].lower() in valid_extensions

# 📦 Secure Archiver
def archive_image(file_path):
    base = os.path.basename(file_path)
    target = os.path.join(ARCHIVE_FOLDER, base)
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    try:
        with open(file_path, 'rb') as src, open(target, 'wb') as dst:
            dst.write(src.read())
        log_event(f"Archived: {base}")
    except Exception as e:
        log_event(f"Archive error: {e}", encrypt=True)

# 🕵️‍♂️ Folder Watcher
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and validate_image(event.src_path):
            log_event(f"Detected new image: {event.src_path}")
            archive_image(event.src_path)

def start_watcher():
    observer = Observer()
    observer.schedule(ImageHandler(), WATCH_FOLDER, recursive=False)
    observer.start()
    log_event("Watcher started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 🎨 Basic GUI
def run_gui():
    root = Tk()
    root.title("MagicBox Sentinel")
    root.geometry("300x150")

    Label(root, text="Monitoring folder:").pack(pady=5)
    Label(root, text=WATCH_FOLDER).pack()

    def start():
        threading.Thread(target=start_watcher, daemon=True).start()
        messagebox.showinfo("Sentinel", "Watcher is running!")

    Button(root, text="Start Watcher", command=start).pack(pady=10)
    root.mainloop()

# 🧠 Entry Point
if __name__ == "__main__":
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    run_gui()

