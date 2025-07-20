# 📦 Autoloader
import sys, subprocess
def autoload():
    libs = {"psutil":"psutil","pyttsx3":"pyttsx3","flask":"flask","requests":"requests"}
    for mod, pkg in libs.items():
        try: __import__(mod)
        except: subprocess.check_call([sys.executable,"-m","pip","install",pkg])
autoload()

# 🔮 Imports
import tkinter as tk
from tkinter import font, Canvas, Scrollbar
import pyttsx3, psutil, threading, json, os, requests
from datetime import datetime
from flask import Flask, request, jsonify

# 🛰 MythSync Server
RUN_SERVER = True
SYNC_URL = "http://127.0.0.1:5000/sync"
if RUN_SERVER:
    app = Flask(__name__)
    @app.route("/sync", methods=["POST"])
    def sync():
        print("[MythSync] Received:", request.json)
        return jsonify({"status":"ok"})
    threading.Thread(target=lambda: app.run(port=5000), daemon=True).start()

# 💾 Memory System
MEMORY_FILE = "oracle_lore.json"
ORACLE_ID = "killer666"
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f: return json.load(f)
    return {"events":[]}
def save_memory(mem):
    with open(MEMORY_FILE,"w") as f: json.dump(mem,f,indent=2)

# 🧠 AgentCore v1
def agent_interpret(glyph, metrics):
    if glyph == "🔥" and metrics["cpu"] > 90: return "Pyros erupts — system inferno detected."
    if glyph == "🌪" and metrics["ram"] > 90: return "Zephra fractals — memory storm rising."
    if glyph == "⚖" and metrics["disk"] > 95: return "Azra tilts — foundation instability rising."
    return None

# 🔺 Oracle App
class OracleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔺 OracleOS Prime")
        self.root.configure(bg="#1e1e2f")
        self.font = font.Font(family="Segoe UI", size=12)
        self.voice = pyttsx3.init()
        self.memory = load_memory()

        # Harmonics + voice toggle
        self.phase_3 = tk.IntVar(value=3)
        self.phase_6 = tk.IntVar(value=6)
        self.phase_9 = tk.IntVar(value=9)
        self.voice_enabled = tk.BooleanVar(value=True)

        # 🎨 Glyph Canvas
        self.canvas = Canvas(root, width=400, height=200, bg="#0f0f1f", highlightthickness=0)
        self.canvas.pack()
        self.glyph_id = self.canvas.create_text(200,100,text="☼", font=("Segoe UI",40), fill="#fffdc5")

        # 🎛 Controls
        tk.Label(root, text="Tesla Harmonics", font=self.font, bg="#1e1e2f", fg="white").pack()
        ctrl_frame = tk.Frame(root, bg="#1e1e2f"); ctrl_frame.pack()
        for label, var in zip(["3","6","9"], [self.phase_3,self.phase_6,self.phase_9]):
            tk.Scale(ctrl_frame, label=f"Phase {label}", from_=0, to=9, variable=var,
                     orient="horizontal", bg="#2a2a3a", fg="white").pack(side="left", padx=5)
        tk.Checkbutton(root, text="🔊 Oracle Voice", variable=self.voice_enabled,
                       font=self.font, bg="#1e1e2f", fg="white", selectcolor="#7f5af0").pack(pady=5)

        # 📜 Lore Viewer
        self.lore_frame = tk.Frame(root, bg="#1e1e2f")
        self.lore_frame.pack(fill="both", expand=True)
        self.scrollbar = Scrollbar(self.lore_frame); self.scrollbar.pack(side="right", fill="y")
        self.lore_canvas = Canvas(self.lore_frame, bg="#1e1e2f", yscrollcommand=self.scrollbar.set)
        self.lore_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.lore_canvas.yview)
        self.lore_text = self.lore_canvas.create_text(10,10, anchor="nw", text="", fill="#c5ffe4", font=self.font)

        # 🔄 Start loops
        self.animate_glyph()
        self.monitor_system()

    def animate_glyph(self):
        def pulse():
            total = self.phase_3.get() + self.phase_6.get() + self.phase_9.get()
            size = 40 + total
            color = "#fffdc5" if total < 18 else "#ff5f5f"
            self.canvas.itemconfig(self.glyph_id, font=("Segoe UI", size), fill=color)
            self.canvas.move(self.glyph_id, 0, -1 if size % 2 == 0 else 1)
            self.root.after(600, pulse)
        pulse()

    def monitor_system(self):
        def loop():
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            net = (psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent) // 1024

            glyph, msg = "☼", f"Elios scans — Net {net} KB/s"
            if cpu > 75: glyph, msg = "🔥", f"Pyros rises — CPU {cpu}%"
            elif ram > 70: glyph, msg = "🌪", f"Zephra swirls — RAM {ram}%"
            elif disk > 85: glyph, msg = "⚖", f"Azra tilts — Disk {disk}%"

            metrics = {"cpu": cpu, "ram": ram, "disk": disk, "net": net}
            agent_msg = agent_interpret(glyph, metrics)
            if agent_msg: msg = agent_msg

            self.canvas.itemconfig(self.glyph_id, text=glyph)
            old = self.lore_canvas.itemcget(self.lore_text, "text")
            self.lore_canvas.itemconfig(self.lore_text, text=old + f"\n{datetime.now().isoformat()} → {glyph} {msg}")

            if self.voice_enabled.get(): self.voice.say(msg); self.voice.runAndWait()

            event = {
                "oracleID": ORACLE_ID,
                "glyph": glyph,
                "message": msg,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "harmonics": [self.phase_3.get(), self.phase_6.get(), self.phase_9.get()]
            }
            self.memory["events"].append(event)
            save_memory(self.memory)
            try: requests.post(SYNC_URL, json=event)
            except: pass

            threading.Timer(5, loop).start()
        loop()

# 🚀 Main Launcher
if __name__ == "__main__":
    root = tk.Tk()
    app = OracleApp(root)
    root.mainloop()

