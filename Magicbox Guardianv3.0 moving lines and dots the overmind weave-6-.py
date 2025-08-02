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

# 🔐 Configs & state
trusted_ports = [80, 443, 22, 53]
glyph_trace_memory = []
voice_enabled = True

distortion_colors = {
    "high_entropy": "Crimson Pulse",
    "unsigned_payload": "Void Surge",
    "rejected": "Shadow Fracture"
}
persona_archetypes = {
    "ghost": "Silent Echo",
    "firebrand": "Glyph Burn",
    "oracle": "Temporal Whisper"
}

# 🎙️ Voice Engine
voice = pyttsx3.init()
voice.setProperty("rate", 160)
voice.setProperty("volume", 0.9)

def speak_overmind(text):
    global voice_enabled
    print(f"🎙️ Overmind Speaks: {text}")
    if voice_enabled:
        voice.say(text)
        voice.runAndWait()

# 💠 Node Class (animated glyphs)
class Node:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.radius = 3
        self.intensity = 0

    def energize(self, strength):
        self.intensity = min(100, self.intensity + strength)

    def fade(self):
        self.intensity = max(0, self.intensity - 2)

    def move(self, width, height):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= width: self.dx *= -1
        if self.y <= 0 or self.y >= height: self.dy *= -1

    def draw(self):
        self.fade()
        glow = int(255 * (self.intensity / 100))
        color = f"#00F7{glow:02X}"
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill=color, outline=""
        )

# 🖥️ GUI Class
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

        self.console = tk.Text(root, width=40, bg="black", fg="lime",
                               font=("Consolas", 10))
        self.console.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.status = ttk.Label(root, text="Glyphstream initiated...", anchor="w")
        self.status.pack(fill=tk.X)

        # 🔊 Speaker Icon Button
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

# 🌐 Node Energy Activation
def energize_glyph_nodes(src_ip):
    index = hash(src_ip) % len(gui.nodes)
    for i in range(index, index + 3):
        gui.nodes[i % len(gui.nodes)].energize(random.randint(30, 70))

# 🔍 Packet Handler
def evaluate_packet(packet):
    try:
        src = packet[scapy.IP].src
        dst = packet[scapy.IP].dst
        port = packet[scapy.TCP].dport if packet.haslayer(scapy.TCP) else None
        entropy = random.uniform(0, 1)
        unsigned = "X-Signature" not in str(packet)

        trace = {
            "src": src, "dst": dst, "port": port,
            "entropy": entropy, "unsigned": unsigned,
            "time": time.strftime("%H:%M:%S")
        }

        glyph_trace_memory.append(trace)

        if port not in trusted_ports or entropy > 0.7 or unsigned:
            color = distortion_colors["high_entropy"] if entropy > 0.7 else (
                distortion_colors["unsigned_payload"] if unsigned else distortion_colors["rejected"]
            )
            trigger_persona(trace)
            energize_glyph_nodes(trace['src'])
            gui.log(f"🌀 {color} :: {trace}")

    except Exception as e:
        logging.error(f"Packet error: {e}")

# 🎭 Persona Confrontation
def trigger_persona(trace):
    threat = "firebrand" if trace["entropy"] > 0.8 else ("oracle" if trace["unsigned"] else "ghost")
    persona = persona_archetypes[threat]
    line = f"⚠️ {persona} activated — glyph echo from {trace['src']} to {trace['dst']}"
    gui.log(line)
    speak_overmind(line)

# 🛡️ Process Monitor
def monitor_processes():
    while True:
        for proc in psutil.process_iter(attrs=['pid', 'name']):
            if proc.info['name'] not in ['systemd', 'python']:
                gui.log(f"👁️ Unexpected Process: {proc.info}")
        time.sleep(10)

# 📡 Sniffer Thread
def start_sniffer():
    gui.log("🔮 Overmind Glyphstream Activated...")
    scapy.sniff(filter="ip", prn=evaluate_packet)

# 🚀 Launch Sequence
def launch_overmind():
    threading.Thread(target=start_sniffer, daemon=True).start()
    threading.Thread(target=monitor_processes, daemon=True).start()
    root.mainloop()

# 🌀 Entry Point
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    root = tk.Tk()
    gui = MagicBoxGUI(root)
    gui.log("🌌 Overmind Online. Neural lattice syncing...")
    launch_overmind()

