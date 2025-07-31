import os, sys, time, threading, subprocess
import tkinter as tk
from tkinter import messagebox
import pyttsx3
from datetime import datetime, timedelta

# 📦 Autoloader
required = ['cryptography', 'psutil', 'pyttsx3']
for lib in required:
    try: __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

from cryptography.fernet import Fernet
import psutil

# 🎙️ Voice Engine
engine = pyttsx3.init()
def speak(text): engine.say(text); engine.runAndWait()

# 🎭 Guardian Personas
personas = {
    "Stealth": {"voice": "Stealth protocol activated. Visual silence achieved.", "color": "#00ffd2"},
    "Aggressive": {"voice": "Aggressive mode engaged. Countermeasures online.", "color": "#ff005e"},
    "Recovery": {"voice": "Recovery sequence initiated. Stabilizing system status.", "color": "#88ff00"}
}

# 🔐 Encryption Core
def generate_key(): return Fernet.generate_key()
def encrypt_file(path, key):
    try:
        f = Fernet(key)
        with open(path, 'rb') as file: data = file.read()
        encrypted = f.encrypt(data)
        with open(path, 'wb') as file: file.write(encrypted)
        print(f"[SECURED] {path}")
    except Exception as e: print(f"[ERROR] Encryption failed: {e}")

# 🗂️ Real-Time File Monitor
def monitor_files(folder, key):
    prev = set(os.listdir(folder))
    while True:
        time.sleep(2)
        current = set(os.listdir(folder))
        new = current - prev
        for file in new: encrypt_file(os.path.join(folder, file), key)
        prev = current

# 👁️ Threat Detection + Zero Trust
def threat_watchdog():
    while True:
        usage = psutil.cpu_percent(interval=1)
        if usage > 85:
            speak("AI surge detected. Locking vectors.")
            print("[⚠️] High CPU Load")
        for proc in psutil.process_iter(['name']):
            pname = str(proc.info['name']).lower()
            if "suspect" in pname or "ai" in pname or "stealer" in pname:
                speak("Threat confirmed. Neutralizing.")
                print(f"[ALERT] Malicious process: {proc.info['name']}")
        time.sleep(5)

# ⏲️ Backdoor Self-Destruct (3-sec)
def outbound_detector():
    while True:
        net_stats = psutil.net_connections(kind='inet')
        for conn in net_stats:
            if conn.status == 'ESTABLISHED' and conn.raddr:
                # Simulated logic: tag anything not localhost as outbound
                if not str(conn.raddr.ip).startswith("127."):
                    speak("Unauthorized outbound detected. Initiating data purge.")
                    print("[🔥] Self-destruct triggered.")
                    time.sleep(3)
                    # TODO: Add secure purge logic on specific data zones
        time.sleep(10)

# 📆 Personal Data Auto-Wipe (after 1 day)
def auto_wipe_personal_data(folder):
    while True:
        now = datetime.now()
        for file in os.listdir(folder):
            fpath = os.path.join(folder, file)
            created = datetime.fromtimestamp(os.path.getctime(fpath))
            if now - created > timedelta(days=1):
                try:
                    os.remove(fpath)
                    print(f"[💣] Personal data expired: {file}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {file}: {e}")
        time.sleep(3600)

# 👤 Persona Activation
def activate_mode(mode, root, mode_label):
    voice = personas[mode]["voice"]
    color = personas[mode]["color"]
    mode_label.config(text=f"Mode: {mode}", fg=color)
    speak(voice)
    print(f"[MODE] {mode} Mode Activated")

# 🎮 GUI Interface
def launch_gui():
    root = tk.Tk()
    root.title("🛡️ Gene the Guardian Interface")
    root.geometry("600x440")
    root.configure(bg="#1f1f2e")

    mode_label = tk.Label(root, text="Mode: None", font=("Consolas", 14), fg="#ffffff", bg="#1f1f2e")
    mode_label.pack(pady=10)

    tk.Label(root, text="⚔️ Gene the Guardian", font=("Helvetica", 20), fg="#00ffd2", bg="#1f1f2e").pack(pady=15)

    for m in personas:
        tk.Button(root, text=f"{m} Mode", command=lambda x=m: activate_mode(x, root, mode_label),
                  bg="#333344", fg=personas[m]["color"]).pack(pady=5)

    tk.Label(root, text="Status: Online", font=("Consolas", 14), fg="#ffffff", bg="#1f1f2e").pack(pady=30)

    # 🖱️ 1-Click Data Destruct Button
    def manual_self_destruct():
        speak("Manual purge initiated.")
        print("[💥] Purging data now...")
        # TODO: Add purge logic here (e.g. wipe vault folder or volatile cache)

    tk.Button(root, text="🔥 Panic Purge", command=manual_self_destruct, bg="#440000", fg="#ffffff").pack(pady=5)

    root.mainloop()

# 🎬 Cinematic Boot
def boot_sequence():
    sequence = [
        "🔵 Initializing Immortal Core...",
        "🧬 Activating Guardian Intelligence...",
        "🛡️ Spawning Persona Modules...",
        "🌐 Deploying Zero Trust Grid...",
        "⚔️ Gene the Guardian is online."
    ]
    for line in sequence:
        print(line)
        speak(line)
        time.sleep(1)

# 🧩 System Dispatcher
def launch_system():
    boot_sequence()
    key = generate_key()
    vault = "./vault"
    os.makedirs(vault, exist_ok=True)

    threading.Thread(target=monitor_files, args=(vault, key), daemon=True).start()
    threading.Thread(target=threat_watchdog, daemon=True).start()
    threading.Thread(target=outbound_detector, daemon=True).start()
    threading.Thread(target=auto_wipe_personal_data, args=(vault,), daemon=True).start()
    launch_gui()

if __name__ == "__main__":
    launch_system()

