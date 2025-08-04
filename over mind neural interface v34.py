import tkinter as tk
import asyncio
import threading
import random
import math
from datetime import datetime, timedelta

# === CONFIG ===
PERSONAL_DATA_TTL = timedelta(days=1)
BACKDOOR_DATA_SELF_DESTRUCT_DELAY = 3  # seconds
THREAT_LEVEL = 0.0  # dynamic

# === Async Data Packet ===
class AsyncDataPacket:
    def __init__(self, content, is_personal=False, is_backdoor=False):
        self.content = content
        self.is_personal = is_personal
        self.is_backdoor = is_backdoor
        self.timestamp = datetime.now()

    async def self_destruct(self):
        delay = BACKDOOR_DATA_SELF_DESTRUCT_DELAY if self.is_backdoor else PERSONAL_DATA_TTL.total_seconds()
        await asyncio.sleep(delay)
        print(f"💣 Data Destroyed: {self.content}")
        self.content = None

# === Async Threat Monitor ===
class ThreatMonitor:
    async def monitor(self):
        global THREAT_LEVEL
        while True:
            THREAT_LEVEL = random.uniform(0, 1)
            if THREAT_LEVEL > 0.8:
                print("🛡️ ASI Defense Triggered!")
            await asyncio.sleep(5)

# === Async Persona Pulse ===
class HoloPersona:
    def __init__(self, name, emotion="Calm", intensity=0.5):
        self.name = name
        self.emotion = emotion
        self.intensity = intensity

    async def pulse(self):
        while True:
            print(f"🔮 {self.name} persona → {self.emotion} ({self.intensity})")
            await asyncio.sleep(self.intensity)

# === LCARS Interface ===
class LCARSInterface:
    def display(self, persona):
        print("🔷 LCARS Dashboard")
        print(f"Persona: {persona.name} [{persona.emotion}]")
        print(f"Threat Level: {THREAT_LEVEL}")

# === GUI Node Visuals ===
class Node:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.radius = 3
        self.base_color = "#00F7FF"

    def move(self, width, height):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= width:
            self.dx *= -1
        if self.y <= 0 or self.y >= height:
            self.dy *= -1

    def draw(self, threat):
        flare_color = "#FF0000" if threat > 0.8 else self.base_color
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill=flare_color, outline=""
        )

# === GUI Initialization ===
def launch_gui(async_loop):
    root = tk.Tk()
    root.title("🧠 Overmind Neural Interface")
    root.geometry("720x520")
    root.configure(bg="#0B0E1A")

    canvas_width = 700
    canvas_height = 460
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height,
                       bg="#0A0C1B", highlightthickness=0)
    canvas.pack(pady=30)

    node_count = 40
    nodes = [Node(canvas, canvas_width, canvas_height) for _ in range(node_count)]

    def animate():
        canvas.delete("all")
        for node in nodes:
            node.move(canvas_width, canvas_height)
            node.draw(THREAT_LEVEL)
        for i in range(node_count):
            for j in range(i + 1, node_count):
                n1, n2 = nodes[i], nodes[j]
                dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                if dist < 150:
                    color = "#FF0000" if THREAT_LEVEL > 0.8 else "#00F7FF"
                    canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill=color, width=1)
        root.after(30, animate)

    animate()
    root.mainloop()

# === Async Loop Management ===
def run_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# === Boot Overmind Engine ===
def overmind():
    # Async loop thread
    loop = asyncio.new_event_loop()
    threading.Thread(target=run_async_loop, args=(loop,), daemon=True).start()

    # Create persona + threat
    persona = HoloPersona("Echo", emotion="Focused", intensity=0.7)
    threat_monitor = ThreatMonitor()

    # Schedule async tasks
    asyncio.run_coroutine_threadsafe(persona.pulse(), loop)
    asyncio.run_coroutine_threadsafe(threat_monitor.monitor(), loop)
    asyncio.run_coroutine_threadsafe(AsyncDataPacket("Sensitive", is_personal=True).self_destruct(), loop)
    asyncio.run_coroutine_threadsafe(AsyncDataPacket("Backdoor payload", is_backdoor=True).self_destruct(), loop)

    # Launch GUI
    launch_gui(loop)

# 🚀 Execute Overmind
if __name__ == "__main__":
    overmind()

