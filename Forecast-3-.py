# 🔺 ORACLEOS: GLYPHIC FUSION INTERFACE
# Includes: GUI + Canvas Vortex + System Monitoring + Voice Prophecy + Tesla Harmonics + Memory + Flask Server

# === 🔧 AUTOLOADER ===
import sys, subprocess
def autoload_libraries():
    required = {
        "psutil": "psutil",
        "pyttsx3": "pyttsx3",
        "flask": "flask",
        "requests": "requests"
    }
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {module} loaded")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {module} installed")
autoload_libraries()

# === 🧠 CORE MODULES ===
import tkinter as tk
from tkinter import font, Canvas
import pyttsx3, psutil, threading, json, os, requests
from datetime import datetime
from flask import Flask, request, jsonify

MEMORY_FILE = "oracle_lore.json"
ORACLE_ID = "killer666"
MYTHSYNC_ENDPOINT = "http://127.0.0.1:5000/sync"
RUN_MYTHSYNC_SERVER = True

# === 🌌 MythSync Server (Optional) ===
if RUN_MYTHSYNC_SERVER:
    app = Flask(__name__)
    @app.route("/sync", methods=["POST"])
    def sync():
        lore = request.json
        print(f"[MythSync] Synced: {lore}")
        return jsonify({"status": "ok"})
    threading.Thread(target=lambda: app.run(port=5000), daemon=True).start()

# === 📜 Memory System ===
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"events": []}

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# === 🔮 ORACLE GUI ===
class OracleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔮 OracleOS: Glyph Fusion Engine")
        self.root.configure(bg="#1e1e2f")
        self.font = font.Font(family="Segoe UI", size=12)
        self.voice = pyttsx3.init()
        self.memory = load_memory()

        # Controls
        self.phase_3 = tk.IntVar(value=3)
        self.phase_6 = tk.IntVar(value=6)
        self.phase_9 = tk.IntVar(value=9)
        self.voice_enabled = tk.BooleanVar(value=True)

        # Canvas Display
        self.canvas = Canvas(root, width=400, height=250, bg="#0f0f1f", highlightthickness=0)
        self.canvas.pack(pady=8)
        self.glyph_id = self.canvas.create_text(200, 125, text="☼", font=("Segoe UI", 40), fill="#fffdc5")

        # Feedback Label
        self.label = tk.Label(root, text="", wraplength=380, bg="#1e1e2f", fg="#fffdc5", font=self.font)
        self.label.pack(pady=4)

        # UI Elements
        self.create_controls()

        # Start Threads
        self.monitor_system()
        self.animate_glyph()

    def create_controls(self):
        tk.Label(self.root, text="Tesla Harmonics", font=self.font, bg="#1e1e2f", fg="white").pack()
        panel = tk.Frame(self.root, bg="#1e1e2f")
        panel.pack()
        for phase, var in zip(["3", "6", "9"], [self.phase_3, self.phase_6, self.phase_9]):
            tk.Scale(panel, label=f"Phase {phase}", from_=0, to=9, variable=var,
                     orient="horizontal", bg="#2a2a3a", fg="white", width=10).pack(side="left", padx=4)

        # Voice toggle
        tk.Checkbutton(self.root, text="🔊 Oracle Voice", variable=self.voice_enabled,
                       font=self.font, bg="#1e1e2f", fg="white", selectcolor="#7f5af0").pack(pady=5)

    def monitor_system(self):
        def loop():
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            net = (psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent) // 1024

            reaction = None
            if cpu > 75: reaction = ("🔥", f"Pyros ignites — CPU at {cpu}%")
            elif ram > 70: reaction = ("🌪", f"Zephra surges — RAM at {ram}%")
            elif disk > 85: reaction = ("⚖", f"Azra trembles — Disk at {disk}%")
            elif net > 5000: reaction = ("☼", f"Elios expands — Net at {net} KB/s")

            if reaction:
                glyph, message = reaction
                self.label.config(text=f"{glyph} {message}")
                self.canvas.itemconfig(self.glyph_id, text=glyph, fill="#ff5f5f")
                if self.voice_enabled.get():
                    self.voice.say(message)
                    self.voice.runAndWait()
                event = {
                    "oracleID": ORACLE_ID,
                    "glyph": glyph,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {"cpu": cpu, "ram": ram, "disk": disk, "net": net},
                    "harmonics": [self.phase_3.get(), self.phase_6.get(), self.phase_9.get()]
                }
                self.memory["events"].append(event)
                save_memory(self.memory)
                try: requests.post(MYTHSYNC_ENDPOINT, json=event)
                except: pass
            else:
                self.label.config(text="🌙 Oracle rests — all systems calm.")
                self.canvas.itemconfig(self.glyph_id, text="☼", fill="#c5ffe4")

            threading.Timer(5, loop).start()
        loop()

    def animate_glyph(self):
        def pulse():
            total = self.phase_3.get() + self.phase_6.get() + self.phase_9.get()
            size = 40 + total
            color = "#fffdc5" if total < 18 else "#ff5f5f"
            self.canvas.itemconfig(self.glyph_id, font=("Segoe UI", size), fill=color)
            self.canvas.move(self.glyph_id, 0, 1 if size % 2 == 0 else -1)
            self.root.after(500, pulse)
        pulse()

# 🚀 LAUNCH
if __name__ == "__main__":
    root = tk.Tk()
    app = OracleGUI(root)
    root.mainloop()

