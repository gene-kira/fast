import os, sys, time, threading, subprocess
import tkinter as tk
from tkinter import messagebox
import pyttsx3
from datetime import datetime, timedelta
import psutil

# 📦 Autoloader for Required Libraries
required = ['cryptography', 'psutil', 'pyttsx3']
for lib in required:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

from cryptography.fernet import Fernet

# 🔊 Voice Toggle and Timer
voice_enabled = True
voice_muted_since = None
quiet_timer_seconds = 60  # Auto-reactivation after 60 seconds

# 🎙️ Voice Engine
engine = pyttsx3.init()
def speak(text):
    global voice_enabled, voice_muted_since
    if not voice_enabled and voice_muted_since:
        if time.time() - voice_muted_since >= quiet_timer_seconds:
            voice_enabled = True
            voice_muted_since = None
            print("[VOICE] Auto-reactivated.")
    if voice_enabled:
        engine.say(text)
        engine.runAndWait()

# 🎭 Guardian Personas
personas = {
    "Stealth": {"voice": "Stealth protocol activated. Visual silence achieved.", "color": "#00ffd2"},
    "Aggressive": {"voice": "Aggressive mode engaged. Countermeasures online.", "color": "#ff005e"},
    "Recovery": {"voice": "Recovery sequence initiated. Stabilizing system status.", "color": "#88ff00"}
}

# 🔐 AES-like Encryption Engine
def generate_key(): return Fernet.generate_key()

def encrypt_file(path, key):
    try:
        f = Fernet(key)
        with open(path, 'rb') as file: data = file.read()
        encrypted = f.encrypt(data)
        with open(path, 'wb') as file: file.write(encrypted)
        print(f"[SECURED] {path}")
    except Exception as e: print(f"[ERROR] Failed to encrypt {path}: {e}")

# 🗂️ Real-Time File Monitoring
def monitor_files(folder, key):
    prev = set(os.listdir(folder))
    while True:
        time.sleep(2)
        current = set(os.listdir(folder))
        new = current - prev
        for file in new:
            encrypt_file(os.path.join(folder, file), key)
        prev = current

# 👁️ AI/ASI/Hacker Threat Detection
def threat_watchdog():
    while True:
        usage = psutil.cpu_percent(interval=1)
        if usage > 85:
            speak("AI surge detected. Defensive grid activated.")
            print("[⚠️] High CPU usage")
        for proc in psutil.process_iter(['name']):
            pname = str(proc.info['name']).lower()
            if "suspect" in pname or "stealer" in pname or "ai" in pname:
                speak("Malicious process detected and flagged.")
                print(f"[ALERT] Suspicious process: {proc.info['name']}")
        time.sleep(5)

# ⏲️ Backdoor Leakage Self-Destruct Protocol (3-sec Trigger)
def outbound_detector():
    while True:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.raddr:
                ip = str(conn.raddr.ip)
                if not ip.startswith("127.") and not ip.startswith("192.168"):
                    speak("Outbound leak detected. Initiating purge.")
                    print("[🔥] Unauthorized transmission — purge in 3 seconds")
                    time.sleep(3)
                    # TODO: Add secure wipe logic for volatile zones
        time.sleep(10)

# 📆 Personal Data Auto-Wipe (after 24 hours)
def auto_wipe_personal_data(folder):
    while True:
        now = datetime.now()
        for file in os.listdir(folder):
            fpath = os.path.join(folder, file)
            try:
                created = datetime.fromtimestamp(os.path.getctime(fpath))
                if now - created > timedelta(days=1):
                    os.remove(fpath)
                    print(f"[💣] Expired personal data wiped: {file}")
            except Exception as e:
                print(f"[ERROR] Failed to delete {file}: {e}")
        time.sleep(3600)

# 👤 Persona Mode Activation
def activate_mode(mode, mode_label):
    voice = personas[mode]["voice"]
    color = personas[mode]["color"]
    mode_label.config(text=f"Mode: {mode}", fg=color)
    speak(voice)
    print(f"[MODE] {mode} Mode Activated")

# 🎮 GUI Defense Command Panel
def launch_gui():
    global voice_enabled, voice_muted_since
    root = tk.Tk()
    root.title("🛡️ Gene the Guardian v6.0")
    root.geometry("600x460")
    root.configure(bg="#1f1f2e")

    tk.Label(root, text="⚔️ Gene the Guardian — Immortal Protocol", font=("Helvetica", 20),
             fg="#00ffd2", bg="#1f1f2e").pack(pady=10)

    mode_label = tk.Label(root, text="Mode: None", font=("Consolas", 14), fg="#ffffff", bg="#1f1f2e")
    mode_label.pack(pady=5)

    status_indicator = tk.Label(root, text="🔊 Voice: ON", font=("Consolas", 12),
                                fg="#88ff00", bg="#1f1f2e")
    status_indicator.pack(pady=5)

    def toggle_voice():
        global voice_enabled, voice_muted_since
        voice_enabled = not voice_enabled
        state = "ON" if voice_enabled else "OFF"
        color = "#88ff00" if voice_enabled else "#ff005e"
        status_indicator.config(text=f"🔊 Voice: {state}", fg=color)
        if not voice_enabled:
            voice_muted_since = time.time()
        else:
            voice_muted_since = None
        speak(f"Voice {state.lower()}.")

    for mode in personas:
        tk.Button(root, text=f"{mode} Mode", command=lambda m=mode: activate_mode(m, mode_label),
                  bg="#333344", fg=personas[mode]["color"]).pack(pady=5)

    tk.Button(root, text="🔊 Toggle Voice", command=toggle_voice,
              bg="#222244", fg="#ffffff").pack(pady=5)

    def manual_self_destruct():
        speak("Manual data purge triggered.")
        print("[💥] Panic purge initiated.")
        # TODO: Secure deletion logic (vault, volatile zone, etc.)

    tk.Button(root, text="🔥 Panic Purge", command=manual_self_destruct,
              bg="#440000", fg="#ffffff").pack(pady=10)

    def handle_keypress(event):
        if event.char.lower() == 'm':
            toggle_voice()

    root.bind('<Key>', handle_keypress)
    tk.Label(root, text="Press [M] to mute/unmute voice", font=("Consolas", 10),
             fg="#aaaaaa", bg="#1f1f2e").pack(pady=5)

    tk.Label(root, text="Status: Stable", font=("Consolas", 14),
             fg="#ffffff", bg="#1f1f2e").pack(pady=30)
    root.mainloop()

# 🎬 Cinematic Boot Sequence
def boot_sequence():
    lines = [
        "🔵 Initializing Immortal Core...",
        "🧬 Activating Guardian Intelligence...",
        "🛡️ Spawning Persona Modules...",
        "🌐 Deploying Zero Trust Grid...",
        "⚔️ Gene the Guardian is online."
    ]
    for line in lines:
        print(line)
        speak(line)
        time.sleep(1)

# 🧩 Main System Dispatcher
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

