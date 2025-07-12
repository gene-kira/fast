# =============================================================
# Gene ASI System Sentinel GUI — Unified Control Interface
# =============================================================

# === 🧰 Autoloader ===
import sys, subprocess, importlib
modules = ["tkinter", "pyttsx3", "psutil", "schedule", "numpy",
           "scapy", "watchdog"]
for name in modules:
    try: importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])

# === ✅ Imports ===
import tkinter as tk
from tkinter import messagebox
import pyttsx3, psutil, numpy as np
from scapy.all import sniff
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time, datetime, threading

# === 🎙️ Voice Engine ===
engine = pyttsx3.init()
engine.setProperty("rate", 135)
engine.setProperty("voice", engine.getProperty("voices")[0].id)
engine.setProperty("volume", 1.0)
VOICE_ENABLED = True

def speak(text):
    if VOICE_ENABLED:
        engine.say(text)
        engine.runAndWait()
    print(f"[Gene] {text}")

def toggle_voice():
    global VOICE_ENABLED
    VOICE_ENABLED = not VOICE_ENABLED
    status = "enabled" if VOICE_ENABLED else "disabled"
    speak(f"Voice {status}")
    voice_btn.config(text=f"Voice {'ON' if VOICE_ENABLED else 'OFF'}")

def set_volume(val):
    v = int(val)/100
    engine.setProperty("volume", v)
    if VOICE_ENABLED:
        speak(f"Volume set to {int(val)}")

# === 🔧 Defense Functions ===
class WatchdogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            speak(f"⚠️ File modified: {event.src_path}")

def launch_file_watchdog():
    observer = Observer()
    observer.schedule(WatchdogHandler(), ".", recursive=True)
    observer.start()
    speak("File Watchdog activated.")
    threading.Timer(15, observer.stop).start()

def sniff_packets():
    def packet_alert(pkt):
        if pkt.haslayer("Raw") and b"password" in bytes(pkt["Raw"].load):
            speak("⚠️ Packet alert: Possible credential transmission.")
    speak("Packet Scanner activated.")
    sniff(prn=packet_alert, timeout=15)

def profile_asi():
    flagged = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        mem = proc.info["memory_info"].rss / 1024 / 1024
        name = proc.info["name"]
        if "agent" in name.lower() and mem > 150:
            flagged.append(name)
    if flagged:
        speak(f"⚠️ Rogue ASI detected: {', '.join(flagged)}")
    else:
        speak("ASI agents stable.")

def ritual_defense():
    r = np.random.uniform(0.9, 1.2)
    v = 1.618 * r
    speak(f"Ritual seal activated. Resonance = {v:.4f}")
    speak("Symbolic glyph alignment complete.")

# === GUI Layout ===
root = tk.Tk()
root.title("Gene ASI System Sentinel")
root.geometry("420x540")
root.configure(bg="#0c0c0c")

canvas = tk.Canvas(root, width=220, height=220, bg="#0c0c0c", highlightthickness=0)
canvas.pack()
canvas.create_oval(30, 30, 190, 190, outline="#8C52FF", width=3)
canvas.create_oval(60, 60, 160, 160, outline="#FFD700", width=2)
canvas.create_text(110, 20, text="Gene ASI System Sentinel", fill="#C0C0C0", font=("Helvetica", 12, "bold"))
canvas.create_text(110, 110, text="⟁", fill="#8C52FF", font=("Helvetica", 36, "bold"))

button_frame = tk.Frame(root, bg="#0c0c0c")
button_frame.pack()

def thread_target(func):
    threading.Thread(target=func).start()

tk.Button(button_frame, text="☄️ Activate Gene", command=lambda: speak("Gene awakened."), width=30).pack(pady=3)
tk.Button(button_frame, text="📡 Start Packet Scanner", command=lambda: thread_target(sniff_packets), width=30).pack(pady=3)
tk.Button(button_frame, text="📂 Start File Watchdog", command=lambda: thread_target(launch_file_watchdog), width=30).pack(pady=3)
tk.Button(button_frame, text="🧠 Start ASI Profiler", command=lambda: thread_target(profile_asi), width=30).pack(pady=3)
tk.Button(button_frame, text="🔐 Trigger Defense Ritual", command=ritual_defense, width=30).pack(pady=3)
tk.Button(button_frame, text="📜 View Logs", command=lambda: messagebox.showinfo("Gene Logs", "Logs placeholder..."), width=30).pack(pady=3)

# === Voice Controls ===
voice_btn = tk.Button(root, text="Voice ON", command=toggle_voice, width=10)
voice_btn.pack(pady=5)

tk.Label(root, text="🔊 Volume", fg="white", bg="#0c0c0c").pack()
volume_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume, bg="#0c0c0c", fg="white")
volume_slider.set(100)
volume_slider.pack()

status = tk.StringVar()
status.set("🌀 Interface active. Awaiting ritual input.")
tk.Label(root, textvariable=status, fg="#8C52FF", bg="#0c0c0c", font=("Helvetica", 11)).pack(pady=10)

speak("Gene ASI Control Interface initialized.")
root.mainloop()

