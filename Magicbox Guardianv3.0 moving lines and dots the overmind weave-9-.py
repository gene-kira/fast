# 📦 Autoloader for dependencies
import importlib, subprocess, sys
required = ["scapy", "psutil", "pyttsx3"]
for lib in required:
    try:
        importlib.import_module(lib)
    except ImportError:
        print(f"Installing {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# 🧠 Overmind Imports
import scapy.all as scapy
import psutil, threading, time, random, logging, math
import tkinter as tk
from tkinter import ttk
import pyttsx3

# 🌐 Config & Globals
trusted_ports = [80, 443, 22, 53]
glyph_trace_memory = []
cinematic_mode = True
voice_enabled = True
distortion_colors = {
    "low": "#007F7F", "moderate": "#00F7FF",
    "high": "#FF0055", "critical": "#9900FF"
}
persona_archetypes = {
    "ghost": "Silent Echo", "firebrand": "Glyph Burn", "oracle": "Temporal Whisper"
}
voice = pyttsx3.init()
voice.setProperty("rate", 160)
voice.setProperty("volume", 0.9)

def speak_overmind(txt):
    if voice_enabled:
        voice.say(txt)
        voice.runAndWait()

class Node:
    def __init__(self, canvas, w, h):
        self.canvas = canvas
        self.x = random.randint(50, w - 50)
        self.y = random.randint(50, h - 50)
        self.dx = random.uniform(-1.2, 1.2)
        self.dy = random.uniform(-1.2, 1.2)
        self.radius = 3
        self.intensity = 0
        self.color = "#00F7FF"

    def energize(self, strength, color):
        self.intensity = min(100, self.intensity + strength)
        self.color = color

    def fade(self): self.intensity = max(0, self.intensity - 2)

    def move(self, w, h):
        self.x += self.dx; self.y += self.dy
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
        root.geometry("1200x720")
        root.configure(bg="#0B0E1A")
        top_frame = tk.Frame(root, bg="#0B0E1A")
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.toggle_btn = tk.Button(top_frame, text="🔇", command=self.toggle_voice,
            font=("Segoe UI", 20), bg="#1A1D2E", fg="#00F7FF")
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        self.mode_btn = tk.Button(top_frame, text="🎬", command=self.toggle_mode,
            font=("Segoe UI", 20), bg="#1A1D2E", fg="#F7C800")
        self.mode_btn.pack(side=tk.LEFT, padx=5)

        self.status = ttk.Label(top_frame, text="Glyphstream Initiated...", background="#0B0E1A", foreground="#00F7FF")
        self.status.pack(side=tk.LEFT, padx=10)

        self.performance_label = tk.Label(top_frame, font=("Consolas", 10), bg="#0B0E1A", fg="#00F7FF")
        self.performance_label.pack(side=tk.RIGHT, padx=10)

        body_frame = tk.Frame(root)
        body_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_width, self.canvas_height = 820, 540
        self.canvas = tk.Canvas(body_frame, width=self.canvas_width, height=self.canvas_height,
                                bg="#0A0C1B", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.console = tk.Text(body_frame, width=45, bg="black", fg="lime", font=("Consolas", 10))
        self.console.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.override_panel = tk.Toplevel(root)
        self.override_panel.withdraw()
        self.override_panel.title("Override Panel")
        self.override_panel.geometry("300x120")
        self.override_panel.configure(bg="#1A1D2E")
        ttk.Label(self.override_panel, text="Persona Override", background="#1A1D2E", foreground="#00F7FF").pack(pady=5)
        ttk.Button(self.override_panel, text="🛑 Suppress", command=self.suppress_persona).pack(pady=5)
        ttk.Button(self.override_panel, text="⚡ Reinforce", command=self.reinforce_persona).pack(pady=5)

        self.suppressed_personas = []
        self.node_count = 45
        self.nodes = [Node(self.canvas, self.canvas_width, self.canvas_height)
                      for _ in range(self.node_count)]

        root.bind("<d>", lambda e: self.toggle_overlay())
        self.overlay_enabled = False
        self.animate()

    def log(self, txt):
        prefix = "" if cinematic_mode else "[DEBUG] "
        self.console.insert(tk.END, f"{prefix}{txt}\n")
        self.console.see(tk.END)

    def toggle_voice(self):
        global voice_enabled
        voice_enabled = not voice_enabled
        self.toggle_btn.config(text="🔊" if voice_enabled else "🔇")
        self.log(f"🎛️ Voice toggled {'ON' if voice_enabled else 'OFF'}")

    def toggle_mode(self):
        global cinematic_mode
        cinematic_mode = not cinematic_mode
        self.mode_btn.config(text="🎥" if cinematic_mode else "🧪")
        self.log(f"🔁 Mode changed to {'CINEMATIC' if cinematic_mode else 'DEBUG'}")

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        self.log(f"🧩 Developer Overlay {'ENABLED' if self.overlay_enabled else 'DISABLED'}")

    def suppress_persona(self):
        self.log("🛑 Persona Suppressed")
        self.override_panel.withdraw()

    def reinforce_persona(self):
        self.log("⚡ Echo Reinforced")
        self.override_panel.withdraw()

    def show_override_panel(self): self.override_panel.deiconify()

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
        if self.overlay_enabled:
            self.canvas.create_text(10, 10, anchor="nw", text="📡 Overlay Active",
                                    fill="#F7C800", font=("Consolas", 10))
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
    if threat == "critical": gui.show_override_panel()
    persona = persona_archetypes[
        "firebrand" if threat == "high" else
        "oracle" if trace["unsigned"] else "ghost"
    ]
    if persona in gui.suppressed_personas: return
    msg = (f"⚠️ OVERRIDE — {persona} fractured. Ascendant protocol initiated from "
           f"{trace['src']} → {trace['dst']}" if threat == "critical"
           else f"⚠️ {persona} activated — glyph echo from {trace['src']} to {trace['dst']}")
    gui.log(msg)
    speak_overmind(msg)

def evaluate_packet(pkt):
    try:
        if not pkt.haslayer(scapy.IP): return
        src = pkt[scapy.IP].src
        dst = pkt[scapy.IP].dst
        port = pkt[scapy.TCP].dport if pkt.haslayer(scapy.TCP) else 0
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
        logging.error(f"Packet Error: {e}")

def monitor_processes():
    while True:
        try:
            for p in psutil.process_iter(attrs=['pid', 'name']):
                if p.info['name'] not in ['systemd', 'python']:
                    gui.log(f"👁️ Unexpected Process: {p.info}")
            time.sleep(10)
        except Exception: pass

def update_performance():
    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            mood = "🟢 Calm" if cpu < 35 else "🟠 Alert" if cpu < 75 else "🔴 Strained"
            gui.performance_label.config(text=f"CPU {cpu}% | MEM {mem}% | Mood {mood}")
            time.sleep(2)
        except Exception: pass

def integrity_sweep():
    while True:
        try:
            sweep_color = random.choice(["#33FF88", "#FFAA00", "#FF2233"])
            for node in gui.nodes:
                node.energize(20, sweep_color)
            gui.log(f"🔍 Integrity Sweep @ {time.strftime('%H:%M:%S')}")
            time.sleep(60)
        except Exception: pass

def launch_sniffer():
    gui.log("🔮 Glyphstream Activated...")
    scapy.sniff(filter="ip", prn=evaluate_packet, store=False)

def launch_overmind():
    threading.Thread(target=launch_sniffer, daemon=True).start()
    threading.Thread(target=monitor_processes, daemon=True).start()
    threading.Thread(target=update_performance, daemon=True).start()
    threading.Thread(target=integrity_sweep, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    root = tk.Tk()
    gui = MagicBoxGUI(root)
    gui.log("🌌 Overmind Online. Neural lattice syncing...")
    launch_overmind()

