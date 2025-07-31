import os, sys, time, threading, subprocess
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
import psutil
import random

# 📦 Autoloader
required = ['cryptography', 'psutil', 'pyttsx3']
for lib in required:
    try: __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

from cryptography.fernet import Fernet

# 🌐 Globals (voice disabled permanently)
voice_enabled = False
voice_muted_since = time.time()
quiet_timer_seconds = 60
alert_memory = set()
alert_cooldown = 60  # seconds

# 🎙️ Silenced Voice Engine
def speak(text):
    pass  # Voice permanently disabled

# 🎭 Guardian Personas
personas = {
    "Stealth": {"voice": "Stealth protocol engaged.", "color": "#00ffd2"},
    "Aggressive": {"voice": "Aggressive mode engaged.", "color": "#ff005e"},
    "Recovery": {"voice": "Recovery mode initiated.", "color": "#88ff00"}
}

# 🔐 Encryption & Shredding
def generate_key(): return Fernet.generate_key()

def encrypt_file(path, key):
    try:
        f = Fernet(key)
        with open(path, 'rb') as file: data = file.read()
        encrypted = f.encrypt(data)
        with open(path, 'wb') as file: file.write(encrypted)
        print(f"[SECURED] {path}")
    except Exception as e: print(f"[ERROR] Encryption failed: {e}")

def shred_file(path):
    try:
        length = os.path.getsize(path)
        with open(path, 'wb') as file:
            for _ in range(3):
                file.write(os.urandom(length))
                file.flush()
        os.remove(path)
        print(f"[SHREDDED] {path}")
    except Exception as e:
        print(f"[ERROR] Shred failed: {e}")

# 📁 Real-Time File Monitor
def monitor_files(folder, key):
    prev = set(os.listdir(folder))
    while True:
        time.sleep(2)
        current = set(os.listdir(folder))
        new = current - prev
        for file in new:
            encrypt_file(os.path.join(folder, file), key)
        prev = current

# 🕵️ Threat Detection with Alert Cooldown
def threat_watchdog():
    while True:
        usage = psutil.cpu_percent(interval=1)
        if usage > 85 and "cpu_surge" not in alert_memory:
            speak("AI surge detected. Defensive protocols active.")
            print("[⚠️] High CPU Load")
            alert_memory.add("cpu_surge")
            threading.Timer(alert_cooldown, lambda: alert_memory.discard("cpu_surge")).start()

        for proc in psutil.process_iter(['name']):
            pname = str(proc.info['name']).lower()
            if any(tag in pname for tag in ["suspect", "stealer", "ai"]):
                key = f"proc_{pname}"
                if key not in alert_memory:
                    speak("Malicious process detected.")
                    print(f"[ALERT] {proc.info['name']}")
                    alert_memory.add(key)
                    threading.Timer(alert_cooldown, lambda: alert_memory.discard(key)).start()
        time.sleep(5)

# 🚫 Backdoor Detector + Self-Destruct
def outbound_detector():
    while True:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.raddr:
                ip = str(conn.raddr.ip)
                if not ip.startswith("127.") and not ip.startswith("192.168"):
                    if ip not in alert_memory:
                        speak("Unauthorized outbound signal. Self-destruct initiating.")
                        print("[🔥] Leak detected — purging in 3 seconds.")
                        alert_memory.add(ip)
                        threading.Timer(alert_cooldown, lambda: alert_memory.discard(ip)).start()
                        time.sleep(3)
                        for f in os.listdir("./vault"):
                            shred_file(os.path.join("./vault", f))
        time.sleep(10)

# 📆 Auto-Wipe Personal Data (after 24 hours)
def auto_wipe_personal_data(folder):
    while True:
        now = datetime.now()
        for file in os.listdir(folder):
            fpath = os.path.join(folder, file)
            try:
                created = datetime.fromtimestamp(os.path.getctime(fpath))
                if now - created > timedelta(days=1):
                    shred_file(fpath)
            except Exception as e:
                print(f"[ERROR] Auto-wipe failed: {e}")
        time.sleep(3600)

# 🧬 Emergency Override Handler
def override_emergency():
    speak("Emergency override activated. Shutting down all systems.")
    print("[OVERRIDE] Threads terminated.")
    os._exit(1)

# 🎮 GUI Launcher
def launch_gui():
    global voice_enabled, voice_muted_since
    root = tk.Tk()
    root.title("🛡️ Gene the Guardian v7.0")
    root.geometry("620x480")
    root.configure(bg="#1f1f2e")

    tk.Label(root, text="⚔️ Gene the Guardian — Dominion Protocol", font=("Helvetica", 20),
             fg="#00ffd2", bg="#1f1f2e").pack(pady=10)

    mode_label = tk.Label(root, text="Mode: None", font=("Consolas", 14),
                          fg="#ffffff", bg="#1f1f2e")
    mode_label.pack(pady=5)

    voice_status = tk.Label(root, text="🔊 Voice: OFF", font=("Consolas", 12),
                            fg="#ff005e", bg="#1f1f2e")
    voice_status.pack(pady=5)

    badge = tk.Canvas(root, width=100, height=100, bg="#1f1f2e", highlightthickness=0)
    glow = badge.create_oval(10, 10, 90, 90, fill="#222244")
    badge.pack(pady=10)

    def pulse_badge():
        while True:
            color = random.choice(["#00ffd2", "#ff005e", "#88ff00"])
            badge.itemconfig(glow, fill=color)
            time.sleep(2)

    def toggle_voice():
        global voice_enabled, voice_muted_since
        voice_enabled = not voice_enabled
        state = "ON" if voice_enabled else "OFF"
        voice_status.config(text=f"🔊 Voice: {state}",
                            fg="#88ff00" if voice_enabled else "#ff005e")
        if not voice_enabled:
            voice_muted_since = time.time()
        else:
            voice_muted_since = None
        speak(f"Voice {state.lower()}.")

    def activate_mode(mode):
        mode_label.config(text=f"Mode: {mode}", fg=personas[mode]["color"])
        speak(personas[mode]["voice"])

    def manual_self_destruct():
        speak("Manual purge confirmed.")
        print("[💥] Panic purge triggered.")
        for f in os.listdir("./vault"):
            shred_file(os.path.join("./vault", f))

    for mode in personas:
        tk.Button(root, text=f"{mode} Mode",
                  command=lambda m=mode: activate_mode(m),
                  bg="#333344", fg=personas[mode]["color"]).pack(pady=5)

    tk.Button(root, text="🔊 Toggle Voice", command=toggle_voice,
              bg="#222244", fg="#ffffff").pack(pady=5)

    tk.Button(root, text="🔥 Panic Purge", command=manual_self_destruct,
              bg="#440000", fg="#ffffff").pack(pady=10)

    def key_trigger(event):
        if event.char.lower() == 'm': toggle_voice()
        if event.char.lower() == 'k': override_emergency()

    root.bind('<Key>', key_trigger)

    tk.Label(root, text="Hotkeys: [M] Mute/Unmute  |  [K] Override Kill",
             font=("Consolas", 10), fg="#aaaaaa", bg="#1f1f2e").pack(pady=5)

    tk.Label(root, text="Status: Stable", font=("Consolas", 14),
             fg="#ffffff", bg="#1f1f2e").pack(pady=30)

    threading.Thread(target=pulse_badge, daemon=True).start()
    root.mainloop()

# 🎬 Cinematic Boot
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

