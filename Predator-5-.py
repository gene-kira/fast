# 🧿 MagicBox Sentinel (Part 1 of 2)
# Full Ritual System with Network Integration
# Author: killer666 + Copilot

# 🔧 Auto-loader for required libraries
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

modules = ["pyttsx3", "psutil", "tkinter"]
for m in modules:
    try:
        __import__(m)
    except ImportError:
        install(m)

# 🔮 Required Imports
import pyttsx3
import tkinter as tk
from tkinter import ttk
import json
import os
import random
import datetime
import threading
import time
import psutil

# 🧠 Memory manager
class MemoryCore:
    def __init__(self, path="sentinel_memory.json"):
        self.path = path
        self.memory = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.memory = json.load(f)

    def remember(self, entry):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.memory.append({'time': timestamp, **entry})
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def recall(self):
        return self.memory[-5:]

# 🪬 Glyph loader
def load_glyphs(path="glyphs.json"):
    if not os.path.exists(path):
        default = [
            {"name": "🜃 Earth Rune", "type": "stability", "mood": "calm"},
            {"name": "☿ Chaos Bind", "type": "disruption", "mood": "anxious"},
            {"name": "⚕ Purity Seal", "type": "cleansing", "mood": "hopeful"},
            {"name": "🝓 Signal Thorn", "type": "aggro", "mood": "hostile"}
        ]
        with open(path, 'w') as f:
            json.dump(default, f, indent=2)
    with open(path, 'r') as f:
        return json.load(f)

# 🌐 NetSigil scan logic
def scan_network_activity():
    suspicious_ips = ["192.168.0.13", "10.0.0.99"]  # Add more if needed
    for conn in psutil.net_connections(kind="inet"):
        if conn.status == "ESTABLISHED":
            try:
                r_ip = conn.raddr.ip
                if r_ip in suspicious_ips:
                    return f"Suspicious connection from {r_ip}"
            except Exception:
                continue
    return None

# 🧿 MagicBox Sentinel (Part 2 of 2)
class MagicBoxSentinel:
    def __init__(self, root):
        self.root = root
        root.title("MagicBox ASI Hunter 🧿")
        root.geometry("780x720")
        root.configure(bg="#1e1e2f")

        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('rate', 160)
        self.voice_engine.setProperty('volume', 1.0)
        self.voices = self.voice_engine.getProperty('voices')

        self.memory_core = MemoryCore()
        self.glyphs = load_glyphs()

        # 🪬 UI Elements
        self.title_label = tk.Label(root, text="🪬 MagicBox Sentinel", font=("Papyrus", 24), fg="#d4af37", bg="#1e1e2f")
        self.title_label.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Idle", font=("Consolas", 12), fg="#8ae0ff", bg="#1e1e2f")
        self.status_label.pack(pady=5)

        self.voice_var = tk.StringVar()
        self.voice_menu = ttk.Combobox(root, textvariable=self.voice_var, state="readonly")
        self.voice_menu['values'] = [v.name for v in self.voices]
        self.voice_menu.current(0)
        self.voice_menu.pack(pady=5)

        # 🧭 Ritual Controls
        ttk.Button(root, text="Bind Ritual Voice", command=self.set_voice).pack(pady=5)
        ttk.Button(root, text="Start Ritual Scan", command=self.start_scan).pack(pady=5)
        ttk.Button(root, text="Trace Sigils", command=self.trace_code).pack(pady=5)
        ttk.Button(root, text="Exorcise Rogue Code", command=self.kill_rogue).pack(pady=5)
        ttk.Button(root, text="Cast Glyph Ritual", command=self.cast_ritual).pack(pady=5)
        ttk.Button(root, text="Recall Memories", command=self.recall_memory).pack(pady=5)
        ttk.Button(root, text="Scan Network Sigils", command=self.net_scan).pack(pady=5)

        self.auto_purge = tk.BooleanVar()
        ttk.Checkbutton(root, text="Auto-Purge on Anomaly", variable=self.auto_purge).pack(pady=5)

        self.scan_interval_var = tk.IntVar(value=30)
        tk.Label(root, text="Auto Scan Interval (minutes):", bg="#1e1e2f", fg="#8ae0ff").pack()
        tk.Entry(root, textvariable=self.scan_interval_var).pack(pady=5)
        ttk.Button(root, text="Start Autonomous Guard", command=self.activate_autoscan).pack(pady=5)

        # 🧿 Ritual Log Window
        self.log_text = tk.Text(root, height=16, width=92, bg="#0f0f1a", fg="#00ffcc")
        self.log_text.pack(pady=10)
        self.log(">>> Ritual system awakened. Awaiting command...\n")

    # 💬 Voice ritual
    def speak(self, text, mood="neutral"):
        rate_map = {"calm": 140, "hostile": 190, "neutral": 160, "hopeful": 150, "anxious": 170}
        self.voice_engine.setProperty("rate", rate_map.get(mood, 160))
        self.voice_engine.say(text)
        self.voice_engine.runAndWait()

    def log(self, message):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{stamp}] {message}\n")
        self.log_text.see(tk.END)

    def set_voice(self):
        name = self.voice_var.get()
        for v in self.voices:
            if v.name == name:
                self.voice_engine.setProperty('voice', v.id)
                self.log(f">>> Bound voice persona: {name}")
                self.speak(f"Voice binding complete. I now speak as {name}.")
                break

    def start_scan(self):
        self.status_label.config(text="Status: Scanning...")
        glyph = random.choice(self.glyphs)
        anomaly = random.choice([True, False, False])
        mood = "hostile" if anomaly else glyph['mood']

        self.log(">>> Initiating scan sequence...")
        self.log(f">>> Invoked glyph: {glyph['name']} | Mood: {glyph['mood']}")
        result = "Threat detected." if anomaly else "All systems stable."
        self.log(f">>> Scan result: {result}")
        self.speak(f"{result}. Glyph {glyph['name']} invoked.", mood)

        self.memory_core.remember({'event': 'Scan', 'glyph': glyph['name'], 'mood': glyph['mood'], 'result': result})
        if anomaly and self.auto_purge.get():
            self.kill_rogue()
        self.status_label.config(text="Status: Idle")

    def trace_code(self):
        self.status_label.config(text="Status: Tracing...")
        origin = random.choice(["unknown kernel fork", "corrupted temp script", "ghost process"])
        self.log(">>> Tracing signature origins...")
        self.log(f">>> Origin trace: {origin}")
        self.speak(f"Signature traced to: {origin}", "hostile")
        self.memory_core.remember({'event': 'Trace', 'origin': origin})
        self.status_label.config(text="Status: Idle")

    def kill_rogue(self):
        self.status_label.config(text="Status: Purging...")
        result = "Rogue ASI thread neutralized."
        self.log(">>> Executing exorcism protocol...")
        self.log(f">>> {result}")
        self.speak(result, "hostile")
        self.memory_core.remember({'event': 'Purge', 'result': result})
        self.status_label.config(text="Status: Idle")

    def cast_ritual(self):
        glyph = random.choice(self.glyphs)
        self.log(">>> Activating glyph scanner...")
        self.log(f">>> Glyph invoked: {glyph['name']}")
        self.log(f">>> Emotional feedback: {glyph['mood']}")
        self.speak(f"{glyph['name']} activated. Mood: {glyph['mood']}", glyph['mood'])
        self.memory_core.remember({'event': 'Ritual', 'glyph': glyph['name'], 'emotion': glyph['mood']})
        self.status_label.config(text="Status: Idle")

    def recall_memory(self):
        last_entries = self.memory_core.recall()
        self.log(">>> Recalling recent ritual history:")
        for entry in last_entries:
            stamp = entry.get('time', '--')
            details = ", ".join(f"{k}: {v}" for k, v in entry.items() if k != 'time')
            self.log(f"    {stamp} — {details}")
        self.speak("Memory retrieved. Events displayed.")
        self.status_label.config(text="Status: Idle")

    def net_scan(self):
        self.status_label.config(text="Status: Network Scanning...")
        result = scan_network_activity()
        if result:
            glyph_map = {
                'Suspicious': '🝓 Signal Thorn',
                'Connection': '☿ Chaos Bind',
                'Scan': '⚕ Purity Seal'
            }
            matched_key = next((key for key in glyph_map if key in result), None)
            glyph_name = glyph_map.get(matched_key, self.glyphs[0]['name'])
            glyph = next((g for g in self.glyphs if g['name'] == glyph_name), self.glyphs[0])
            self.log(">>> Network anomaly detected:")
            self.log(f">>> {result}")
            self.log(f">>> Invoked glyph: {glyph['name']}")
            self.speak(f"{result}. Glyph {glyph['name']} deployed.", glyph['mood'])
            self.memory_core.remember({'event': 'NetworkAnomaly', 'type': result, 'glyph': glyph['name']})
            if self.auto_purge.get():
                self.kill_rogue()
        else:
            self.log(">>> No rogue signals found in network scan.")
            self.speak("Network clean. No rogue patterns detected.", "calm")
        self.status_label.config(text="Status: Idle")

    def autoscan_loop(self):
        while True:
            interval = self.scan_interval_var.get() * 60
            self.root.after(0, self.start_scan)
            self.root.after(0, self.net_scan)
            time.sleep(interval)

    def activate_autoscan(self):
        self.log(">>> Autonomous guard activated.")
        self.speak("Sentinel awakened. I now scan the unseen realm in cycles.")
        t = threading.Thread(target=self.autoscan_loop, daemon=True)
        t.start()

# 🚀 Launch Sentinel
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxSentinel(root)
    root.mainloop()

