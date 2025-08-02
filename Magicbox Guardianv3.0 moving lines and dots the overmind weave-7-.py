# 📦 Autoloader for external dependencies
import importlib, subprocess, sys
required_libraries = ["scapy", "psutil", "pyttsx3"]
for lib in required_libraries:
    try:
        importlib.import_module(lib)
    except ImportError:
        print(f"📦 Installing {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# 🌌 Overmind Core
import scapy.all as scapy
import psutil
import threading
import time
import random
import logging
import math
import tkinter as tk
from tkinter import ttk
import pyttsx3

trusted_ports = [80, 443, 22, 53]
glyph_trace_memory = []
voice_enabled = True

distortion_colors = {
    "low": "#007F7F",
    "moderate": "#00F7FF",
    "high": "#FF0055",
    "critical": "#9900FF"
}
persona_archetypes = {
    "ghost": "Silent Echo",
    "firebrand": "Glyph Burn",
    "oracle": "Temporal Whisper"
}

voice = pyttsx3.init()
voice.setProperty("rate", 160)
voice.setProperty("volume", 0.9)
def speak_overmind(text):
    if voice_enabled:
        voice.say(text)
        voice.runAndWait()

class Node:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.dx = random.uniform(-1.5, 1.5)
        self.dy = random.uniform(-1.5, 1.5)
        self.radius = 3
        self.intensity = 0
        self.color = "#00F7FF"

    def energize(self, strength, color):
        self.intensity = min(100, self.intensity + strength)
        self.color = color

    def fade(self):
        self.intensity = max(0, self.intensity - 2)

    def move(self, w, h):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= w: self.dx *= -1
        if self.y <= 0 or self.y >= h: self.dy *= -1

    def draw(self):
        self.fade()
        glow = int(255 * (self.intensity / 100))
        c = f"{self.color[:-2]}{glow:02X}"
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill=c, outline=""
        )

class MagicBoxGUI:
    def __init__(self, root):
        root.title("🧠 Overmind Glyphstream Interface")
        root.geometry("1100x670")
        root.configure(bg="#0B0E1A")

        self.canvas_width = 750
        self.canvas_height = 540
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height,
                                bg="#0A0C1B", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.console = tk.Text(root, width=40, bg="black", fg="lime", font=("Consolas", 10))
        self.console.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.status = ttk.Label(root, text="Glyphstream initiated...", anchor="w")
        self.status.pack(fill=tk.X)

        self.toggle_btn = tk.Button(root, text="🔇", command=self.toggle_voice,
            font=("Segoe UI", 32), width=2, height=1,
            bg="#1A1D2E", fg="#00F7FF",
            activebackground="#00F7FF", activeforeground="#000000",
            relief=tk.RAISED, bd=4)
        self.toggle_btn.place(x=20, y=580)

        self.node_count = 45
        self.nodes = [Node(self.canvas, self.canvas_width, self.canvas_height)
                      for _ in range(self.node_count)]
        self.animate()

    def log(self, text):
        self.console.insert(tk.END, f"{text}\n")
        self.console.see(tk.END)

    def toggle_voice(self):
        global voice_enabled
        voice_enabled = not voice_enabled
        new_icon = "🔊" if voice_enabled else "🔇"
        self.toggle_btn.config(text=new_icon)
        state = "ON" if voice_enabled else "OFF"
        self.log(f"🎛️ Voice toggled {state}")

    def animate(self):
        self.canvas.delete("all")
        for node in self.nodes:
            node.move(self.canvas_width, self.canvas_height)
            node.draw()
        for i in range(self.node_count):
            for j in range(i + 1, self.node_count):
                n1, n2 = self.nodes[i], self.nodes[j]
                dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                if dist < 150:
                    self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00F7FF", width=1)
        self.canvas.after(30, self.animate)

def energize_glyph_nodes(ip, color):
    idx = hash(ip) % len(gui.nodes)
    for i in range(idx, idx + 3):
        gui.nodes[i % len(gui.nodes)].energize(random.randint(30, 70), color)

def get_threat_level(entropy, unsigned, port):
    if entropy > 0.85 and port not in trusted_ports and unsigned:
        return "critical"
    elif entropy > 0.7 or unsigned:
        return "high"
    elif entropy > 0.4:
        return "moderate"
    else:
        return "low"

def trigger_persona(trace, threat):
    persona = persona_archetypes[
        "firebrand" if threat == "high" else
        "oracle" if trace["unsigned"] else "ghost"
    ]
    line = (f"⚠️ OVERRIDE — {persona} fractured. Ascendant protocol initiated from "
            f"{trace['src']} → {trace['dst']}" if threat == "critical"
            else f"⚠️ {persona} activated — glyph echo from {trace['src']} to {trace['dst']}")
    gui.log(line)
    speak_overmind(line)

def evaluate_packet(pkt):
    try:
        src = pkt[scapy.IP].src
        dst = pkt[scapy.IP].dst
        port = pkt[scapy.TCP].dport if pkt.haslayer(scapy.TCP) else None
        entropy = random.uniform(0, 1)
        unsigned = "X-Signature" not in str(pkt)

        trace = {
            "src": src, "dst": dst, "port": port,
            "entropy": entropy, "unsigned": unsigned,
            "time": time.strftime("%H:%M:%S")
        }

        glyph_trace_memory.append(trace)

        level = get_threat_level(entropy, unsigned, port)
        color = distortion_colors[level]
        if level != "low":
            trigger_persona(trace, level)
            energize_glyph_nodes(src, color)
            gui.log(f"🌀 {color} :: {trace}")

    except Exception as e:
        logging.error(f"Packet error: {e}")

def monitor_processes():
    while True:
        for p in psutil.process_iter(attrs=['pid', 'name']):
            if p.info['name'] not in ['systemd', 'python']:
                gui.log(f"👁️ Unexpected Process: {p.info}")
        time.sleep(10)

def start_sniffer():
    gui.log("🔮 Overmind Glyphstream Activated...")
    scapy.sniff(filter="ip", prn=evaluate_packet)

def launch_overmind():
    threading.Thread(target=start_sniffer, daemon=True).start()
    threading.Thread(target=monitor_processes, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    root = tk.Tk()
    gui = MagicBoxGUI(root)
    gui.log("🌌 Overmind Online. Neural lattice syncing...")
    launch_overmind()

