# magicbox_sentinel_ultra.py

import sys, subprocess, importlib, os, time, threading, json
from datetime import datetime, timedelta
from tkinter import Tk, Label, Button, messagebox, StringVar, Frame
from tkinter.ttk import Progressbar, Style, Combobox
import hashlib

# 🔧 Auto-install required packages
required = ['watchdog', 'cryptography', 'pyttsx3']
for pkg in required:
    try: importlib.import_module(pkg)
    except: subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet
import pyttsx3

# 📁 Configuration
SETTINGS_FILE = 'magicbox_config.json'
HASHES_FILE = 'seen_hashes.json'
DEFAULTS = {
    "watch_folder": "input_images",
    "archive_folder": "archived_images",
    "log_file": "sentinel_log.txt",
    "encrypt_logs": True,
    "theme": "dark",
    "voice_persona": "neutral",
    "cleanup_days": 7,
    "dev_mode": True
}

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f: json.dump(DEFAULTS, f, indent=4)
        return DEFAULTS
    with open(SETTINGS_FILE) as f: return json.load(f)

config = load_settings()
for folder in [config["watch_folder"], config["archive_folder"]]:
    os.makedirs(folder, exist_ok=True)

# 🔐 Logging
key = Fernet.generate_key()
cipher = Fernet(key)

def log_event(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{timestamp}] {msg}"
    if config["encrypt_logs"]: entry = cipher.encrypt(entry.encode()).decode()
    with open(config["log_file"], 'a') as f: f.write(entry + "\n")
    if config["dev_mode"]: print(f"> {msg}")

# 🎤 Voice
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(text):
    if config["voice_persona"] == "silent": return
    if config["voice_persona"] == "friendly": text = "Hey there! " + text
    engine.say(text)
    engine.runAndWait()

# 💾 Hash Persistence
def load_hashes():
    return json.load(open(HASHES_FILE)) if os.path.exists(HASHES_FILE) else {}

def save_hashes(data): json.dump(data, open(HASHES_FILE, 'w'))

seen_hashes = load_hashes()

def file_hash(path):
    with open(path, 'rb') as f: return hashlib.md5(f.read()).hexdigest()

# 🧪 Validation
def validate_image(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.bmp']

def classify_context(name):
    name = name.lower()
    if 'scan' in name: return "🧾 Scan"
    if 'map' in name: return "🗺️ Map"
    if 'photo' in name: return "📸 Photo"
    return "📁 File"

# 📦 Archive Logic
def archive_image(path):
    h = file_hash(path)
    if h in seen_hashes:
        log_event(f"Duplicate skipped: {path}")
        return
    seen_hashes[h] = path
    save_hashes(seen_hashes)

    base = os.path.basename(path)
    target = os.path.join(config["archive_folder"], base)
    count = 1
    while os.path.exists(target):
        base = f"{os.path.splitext(base)[0]}_{count}{os.path.splitext(base)[1]}"
        target = os.path.join(config["archive_folder"], base)
        count += 1
    try:
        with open(path, 'rb') as src, open(target, 'wb') as dst: dst.write(src.read())
        mood = classify_context(base)
        log_event(f"Archived: {base} {mood}")
        speak(f"{mood} saved successfully.")
    except Exception as e:
        log_event(f"Archive error: {e}")
        speak("Error during archive.")

# 🧹 Cleanup Logic
def cleanup_expired():
    cutoff = datetime.now() - timedelta(days=config["cleanup_days"])
    for fname in os.listdir(config["watch_folder"]):
        fpath = os.path.join(config["watch_folder"], fname)
        if os.path.isfile(fpath) and datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
            try:
                os.remove(fpath)
                log_event(f"Auto-cleaned: {fname}")
            except Exception as e:
                log_event(f"Cleanup error: {e}")

# 🕵️ Folder Watcher
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and validate_image(event.src_path):
            log_event(f"New file detected: {event.src_path}")
            archive_image(event.src_path)

def start_watcher():
    cleanup_expired()
    observer = Observer()
    observer.schedule(ImageHandler(), config["watch_folder"], recursive=False)
    observer.start()
    log_event("Watcher started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 🎨 GUI
def run_gui():
    root = Tk()
    root.title("MagicBox Sentinel Ultra 🛡️")
    root.geometry("460x320")
    root.resizable(False, False)

    style = Style()
    style.theme_use('clam')
    style.configure("TProgressbar", thickness=8, background='lime')

    bg = "#222" if config["theme"] == "dark" else "#EEE"
    fg = "#EEE" if config["theme"] == "dark" else "#222"
    root.configure(bg=bg)

    header = Label(root, text="MagicBox Sentinel Ultra", font=("Helvetica", 16), bg=bg, fg=fg)
    header.pack(pady=10)

    folder_info = Label(root, text=f"Watching: {config['watch_folder']}", bg=bg, fg=fg)
    folder_info.pack()

    status_text = StringVar(value="Status: idle")
    status_label = Label(root, textvariable=status_text, bg=bg, fg=fg)
    status_label.pack(pady=5)

    progress = Progressbar(root, mode="indeterminate", length=300)
    progress.pack(pady=10)

    persona_frame = Frame(root, bg=bg)
    persona_frame.pack()
    Label(persona_frame, text="Voice Persona:", bg=bg, fg=fg).pack(side="left")
    persona_selector = Combobox(persona_frame, values=["neutral", "friendly", "silent"])
    persona_selector.set(config["voice_persona"])
    persona_selector.pack(side="left")

    theme_frame = Frame(root, bg=bg)
    theme_frame.pack(pady=5)
    Label(theme_frame, text="Theme:", bg=bg, fg=fg).pack(side="left")
    theme_selector = Combobox(theme_frame, values=["dark", "light"])
    theme_selector.set(config["theme"])
    theme_selector.pack(side="left")

    def apply_settings():
        config["voice_persona"] = persona_selector.get()
        config["theme"] = theme_selector.get()
        with open(SETTINGS_FILE, 'w') as f: json.dump(config, f, indent=4)
        speak("Settings updated.")
        messagebox.showinfo("Saved", "Settings applied.")

    def launch_watcher():
        progress.start()
        status_text.set("Status: Monitoring folder...")
        threading.Thread(target=start_watcher, daemon=True).start()
        speak("Sentinel activated.")

    Button(root, text="Start Watcher", command=launch_watcher).pack(pady=5)
    Button(root, text="Save Settings", command=apply_settings).pack(pady=5)

    root.mainloop()

# 🚀 Start
if __name__ == "__main__":
    run_gui()

