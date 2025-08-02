# 🔮 MagicBox Guardian v3.0 — Threaded, Symbolic, Defense-Ready

import sys
import subprocess
import importlib
import tkinter as tk
import random
import math
import threading
import pyttsx3

# ⚙️ Autoload missing libraries
def autoload(packages):
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

autoload(["pyttsx3"])

# 🎤 Voice Engine (threaded)
engine = pyttsx3.init()
engine.setProperty('rate', 155)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def threaded_speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# 🧠 Node class — animated glyph trail particles
class Node:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.radius = 3
        self.color = "#00F7FF"
        self.trail = []

    def move(self, width, height):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= width:
            self.dx *= -1
        if self.y <= 0 or self.y >= height:
            self.dy *= -1
        self.trail.append((self.x, self.y))
        if len(self.trail) > 10:
            self.trail.pop(0)

    def draw(self):
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill=self.color, outline=""
        )
        for i in range(len(self.trail) - 1):
            x1, y1 = self.trail[i]
            x2, y2 = self.trail[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill="#003D7F", width=1)

# 🛡️ GuardianNode — zero-trust defense logic
class GuardianNode:
    def __init__(self):
        self.trust_map = {"localhost": 5}
        self.agent_state = "IDLE"
        self.last_threat = None
        self.distortion_pulse = []

    def evaluate(self, packet):
        source = packet.get('origin', 'unknown')
        trust_level = self.trust_map.get(source, 0)

        if trust_level < 3:
            self.trigger_defense(packet)
        else:
            self.process(packet)

    def trigger_defense(self, packet):
        self.agent_state = "DEFENSE ACTIVE"
        scrambled = self.encode_symbols(packet.get('payload', '???'))
        self.last_threat = scrambled
        self.emit_distortion()
        self.schedule_destruction(packet)

    def encode_symbols(self, data):
        return ''.join(chr((ord(c) + 9) % 256) for c in data)

    def schedule_destruction(self, packet):
        threading.Timer(3.0, lambda: self.nuke(packet)).start()

    def nuke(self, packet):
        threaded_speak("Packet self-destructed")
        print("💥 Destroyed:", packet.get("id", "Unknown"))

    def process(self, packet):
        threaded_speak("Packet approved")
        print("✅ Secure:", packet.get("id", "Unknown"))

    def emit_distortion(self):
        self.distortion_pulse = [(random.randint(100, 600), random.randint(100, 400)) for _ in range(6)]

    def draw_distortion(self, canvas):
        for x, y in self.distortion_pulse:
            r = random.randint(20, 50)
            canvas.create_oval(x - r, y - r, x + r, y + r, outline="#FF0033", width=2)

# 🚀 GUI — Core loop
def launch_network_gui():
    root = tk.Tk()
    root.title("🧠 MagicBox Guardian v3.0")
    root.geometry("720x520")
    root.configure(bg="#0B0E1A")

    canvas_width = 700
    canvas_height = 460
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height,
                       bg="#0A0C1B", highlightthickness=0)
    canvas.pack(pady=30)

    node_count = 40
    nodes = [Node(canvas, canvas_width, canvas_height) for _ in range(node_count)]
    guardian = GuardianNode()

    def animate():
        canvas.delete("all")
        for node in nodes:
            node.move(canvas_width, canvas_height)
            node.draw()

        for i in range(node_count):
            for j in range(i + 1, node_count):
                n1, n2 = nodes[i], nodes[j]
                dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                if dist < 150:
                    canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00F7FF", width=1)

        canvas.create_text(350, 20, text=f"Guardian State: {guardian.agent_state}",
                           fill="#FF5588", font=("Consolas", 12))
        if guardian.last_threat:
            canvas.create_text(350, 40, text="⚠️ Threat Scrambled", fill="#FFCC00", font=("Consolas", 10))

        guardian.draw_distortion(canvas)
        root.after(30, animate)

    threaded_speak("Overmind core activated. Awaiting live packet streams.")
    animate()
    root.mainloop()

# 🧠 Execution
if __name__ == "__main__":
    launch_network_gui()

