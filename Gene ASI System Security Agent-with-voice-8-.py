# =============================================================
# Gene ASI System Sentinel GUI — Part 1 with Ritual Control
# =============================================================

import sys, subprocess, importlib, threading, time, tkinter as tk
from tkinter import messagebox
import pyttsx3, psutil
from scapy.all import sniff
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# === 📦 Autoloader ===
modules = ["tkinter", "pyttsx3", "psutil", "numpy", "scapy", "watchdog"]
for name in modules:
    try: importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])

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
    engine.setProperty("volume", int(val)/100)
    if VOICE_ENABLED: speak(f"Volume set to {val}")

# === 📝 Ritual History ===
change_history = []

def log_change(action):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    change_history.append(f"{timestamp} — {action}")
    status_var.set(f"{action} @ {timestamp}")

def revert_last():
    if change_history:
        last = change_history.pop()
        speak(f"⏪ Reverting: {last}")
    else:
        speak("No rituals to revert.")

def save_changes():
    speak("✅ Rituals saved.")
    log_change("Changes saved")

def fix_rituals():
    speak("🛠️ Fixes applied.")
    log_change("Fixes applied")

# === ⚙️ State Management ===
module_states = { "packet": False, "watchdog": False, "asi": False, "defense": False }
selected_modules = []

def toggle_selection(module):
    if module in selected_modules:
        selected_modules.remove(module)
    else:
        selected_modules.append(module)
    speak(f"Selected: {', '.join(selected_modules)}")

def process_selected(action):
    for mod in selected_modules:
        speak(f"{action} applied to {mod}")
        log_change(f"{action} on {mod}")

# GUI and module logic continues in Part 2...

# === 🌐 Scanner Threads ===
sniffing = False
def sniff_packets():
    while sniffing:
        sniff(prn=scan_packet, timeout=1)

def scan_packet(pkt):
    if pkt.haslayer("Raw") and b"password" in bytes(pkt["Raw"].load):
        speak("⚠️ Packet alert: Possible credential transmission.")

def toggle_packet_scanner():
    global sniffing
    module_states["packet"] = not module_states["packet"]
    state = module_states["packet"]
    packet_btn.config(text=f"📡 Packet Scanner: {'ON' if state else 'OFF'}", bg="green" if state else "red")
    toggle_selection("packet")
    if state:
        sniffing = True
        threading.Thread(target=sniff_packets, daemon=True).start()
        speak("Packet scanner activated.")
    else:
        sniffing = False
        speak("Packet scanner disabled.")
    log_change(f"Packet Scanner toggled {'ON' if state else 'OFF'}")

# === 📂 Watchdog ===
observer = None
class WatchdogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            speak(f"⚠️ File modified: {event.src_path}")

def toggle_file_watchdog():
    global observer
    module_states["watchdog"] = not module_states["watchdog"]
    state = module_states["watchdog"]
    watchdog_btn.config(text=f"📂 File Watchdog: {'ON' if state else 'OFF'}", bg="green" if state else "red")
    toggle_selection("watchdog")
    if state:
        observer = Observer()
        observer.schedule(WatchdogHandler(), ".", recursive=True)
        observer.start()
        speak("File watchdog active.")
    else:
        if observer:
            observer.stop()
            observer.join()
        speak("File watchdog disabled.")
    log_change(f"File Watchdog toggled {'ON' if state else 'OFF'}")

# === 🧠 ASI Profiler ===
def toggle_asi_profiler():
    module_states["asi"] = not module_states["asi"]
    state = module_states["asi"]
    asi_btn.config(text=f"🧠 ASI Profiler: {'ON' if state else 'OFF'}", bg="green" if state else "red")
    toggle_selection("asi")
    if state:
        flagged = []
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                mem = proc.info["memory_info"].rss / 1024 / 1024
                name = proc.info["name"]
                if "agent" in name.lower() and mem > 150:
                    flagged.append(name)
            except: continue
        speak(f"⚠️ Rogue ASI detected: {', '.join(flagged)}" if flagged else "ASI agents stable.")
    else:
        speak("ASI profiler disabled.")
    log_change(f"ASI Profiler toggled {'ON' if state else 'OFF'}")

# === 🔐 Defense Protocol ===
def execute_command(command):
    try: subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        speak(f"Command failed: {e}")

def toggle_defense_protocol():
    module_states["defense"] = not module_states["defense"]
    state = module_states["defense"]
    defense_btn.config(text=f"🔐 Defense Ritual: {'ON' if state else 'OFF'}", bg="green" if state else "red")
    toggle_selection("defense")
    if state:
        execute_command("iptables -A INPUT -p tcp --dport 80 -j DROP")
        speak("Ritual seal engaged. Firewall rule added.")
    else:
        execute_command("iptables -D INPUT -p tcp --dport 80 -j DROP")
        speak("Defense ritual disengaged.")
    log_change(f"Defense Protocol toggled {'ON' if state else 'OFF'}")

# === 🖼️ GUI Assembly ===
root = tk.Tk()
root.title("Gene ASI System Sentinel")
root.geometry("460x620")
root.configure(bg="#0c0c0c")

canvas = tk.Canvas(root, width=220, height=220, bg="#0c0c0c", highlightthickness=0)
canvas.pack(pady=5)
canvas.create_oval(30, 30, 190, 190, outline="#8C52FF", width=3)
canvas.create_oval(60, 60, 160, 160, outline="#FFD700", width=2)
canvas.create_text(110, 20, text="Gene ASI System Sentinel", fill="#C0C0C0", font=("Helvetica", 12, "bold"))
canvas.create_text(110, 110, text="⟁", fill="#8C52FF", font=("Helvetica", 36, "bold"))

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

# === 🔮 Ritual Menu ===
tk.Button(frame, text="💾 Save Ritual", command=save_changes, width=32).pack(pady=4)
tk.Button(frame, text="🛠️ Fix Rituals", command=fix_rituals, width=32).pack(pady=4)
tk.Button(frame, text="⏪ Revert Last", command=revert_last, width=32).pack(pady=4)

tk.Button(frame, text="📜 View Logs", command=lambda: messagebox.showinfo("Gene Logs", "\n".join(change_history[-6:])), width=32).pack(pady=4)

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

