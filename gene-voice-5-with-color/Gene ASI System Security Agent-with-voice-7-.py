# =============================================================
# Gene ASI System Sentinel GUI — Full Ritual Control Interface
# =============================================================

# === 🧰 Autoloader ===
import sys, subprocess, importlib
modules = [
    "tkinter", "pyttsx3", "psutil", "numpy",
    "scapy", "watchdog"
]

for name in modules:
    try: importlib.import_module(name)
    except ImportError:
        print(f"📦 Installing module: {name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])

# === ✅ Imports ===
import tkinter as tk
from tkinter import messagebox
import pyttsx3, psutil, numpy as np
from scapy.all import sniff
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading, time

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
    voice_btn.config(text=f"Voice {'ON' if VOICE_ENABLED else 'OFF'}")
    speak(f"Voice {'enabled' if VOICE_ENABLED else 'disabled'}")

def set_volume(val):
    engine.setProperty("volume", int(val) / 100)
    if VOICE_ENABLED:
        speak(f"Volume set to {val}")

# === ⚙️ State Flags ===
module_states = {
    "packet": False,
    "watchdog": False,
    "asi": False,
    "defense": False
}

# === 📡 Packet Scanner ===
def scan_packet(pkt):
    if pkt.haslayer("Raw") and b"password" in bytes(pkt["Raw"].load):
        speak("⚠️ Packet alert: Possible credential transmission.")

def toggle_packet_scanner():
    module_states["packet"] = not module_states["packet"]
    state = module_states["packet"]
    packet_btn.config(
        text=f"📡 Packet Scanner: {'ON' if state else 'OFF'}",
        bg="green" if state else "red"
    )
    if state:
        speak("Packet scanner activated.")
        threading.Thread(target=lambda: sniff(prn=scan_packet, timeout=15)).start()
    else:
        speak("Packet scanner disabled.")

# === 📂 File Watchdog ===
class WatchdogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            speak(f"⚠️ File modified: {event.src_path}")

def toggle_file_watchdog():
    module_states["watchdog"] = not module_states["watchdog"]
    state = module_states["watchdog"]
    watchdog_btn.config(
        text=f"📂 File Watchdog: {'ON' if state else 'OFF'}",
        bg="green" if state else "red"
    )
    if state:
        observer = Observer()
        observer.schedule(WatchdogHandler(), ".", recursive=True)
        observer.start()
        speak("File watchdog active.")
        threading.Timer(15, observer.stop).start()
    else:
        speak("File watchdog disabled.")

# === 🧠 ASI Profiler ===
def toggle_asi_profiler():
    module_states["asi"] = not module_states["asi"]
    state = module_states["asi"]
    asi_btn.config(
        text=f"🧠 ASI Profiler: {'ON' if state else 'OFF'}",
        bg="green" if state else "red"
    )
    if state:
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
    else:
        speak("ASI profiler disabled.")

# === 🔐 Defense Ritual ===
def toggle_defense_protocol():
    module_states["defense"] = not module_states["defense"]
    state = module_states["defense"]
    defense_btn.config(
        text=f"🔐 Defense Ritual: {'ON' if state else 'OFF'}",
        bg="green" if state else "red"
    )
    if state:
        r = np.random.uniform(0.9, 1.2)
        v = 1.618 * r
        speak(f"Ritual seal engaged. Fractal resonance calibrated: {v:.4f}")
    else:
        speak("Defense ritual disengaged.")

# === 🖼️ GUI Setup ===
root = tk.Tk()
root.title("Gene ASI System Sentinel")
root.geometry("440x580")
root.configure(bg="#0c0c0c")

# === Glyph Canvas ===
canvas = tk.Canvas(root, width=220, height=220, bg="#0c0c0c", highlightthickness=0)
canvas.pack(pady=5)
canvas.create_oval(30, 30, 190, 190, outline="#8C52FF", width=3)
canvas.create_oval(60, 60, 160, 160, outline="#FFD700", width=2)
canvas.create_text(110, 20, text="Gene ASI System Sentinel", fill="#C0C0C0", font=("Helvetica", 12, "bold"))
canvas.create_text(110, 110, text="⟁", fill="#8C52FF", font=("Helvetica", 36, "bold"))

# === Controls Frame ===
frame = tk.Frame(root, bg="#0c0c0c")
frame.pack()

tk.Button(frame, text="☄️ Activate Gene", command=lambda: speak("Gene awakened."), width=32).pack(pady=4)

packet_btn = tk.Button(frame, text="📡 Packet Scanner: OFF", bg="red", command=toggle_packet_scanner, width=32)
packet_btn.pack(pady=4)

watchdog_btn = tk.Button(frame, text="📂 File Watchdog: OFF", bg="red", command=toggle_file_watchdog, width=32)
watchdog_btn.pack(pady=4)

asi_btn = tk.Button(frame, text="🧠 ASI Profiler: OFF", bg="red", command=toggle_asi_profiler, width=32)
asi_btn.pack(pady=4)

defense_btn = tk.Button(frame, text="🔐 Defense Ritual: OFF", bg="red", command=toggle_defense_protocol, width=32)
defense_btn.pack(pady=4)

tk.Button(frame, text="📜 View Logs", command=lambda: messagebox.showinfo("Gene Logs", "Logs placeholder..."), width=32).pack(pady=4)

# === Voice Controls ===
voice_btn = tk.Button(root, text="Voice ON", command=toggle_voice, width=10)
voice_btn.pack(pady=5)

tk.Label(root, text="🔊 Volume", fg="white", bg="#0c0c0c").pack()
slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume, bg="#0c0c0c", fg="white")
slider.set(100)
slider.pack()

status_var = tk.StringVar()
status_var.set("🌀 Interface active. Awaiting ritual input.")
tk.Label(root, textvariable=status_var, fg="#8C52FF", bg="#0c0c0c", font=("Helvetica", 11)).pack(pady=8)

speak("Gene ASI Control Interface initialized.")
root.mainloop()

