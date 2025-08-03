import tkinter as tk
import random
import math
import json
import time

# 🧠 Node Class — Autonomous Entities
class Node:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.radius = 4

    def move(self, width, height):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= width: self.dx *= -1
        if self.y <= 0 or self.y >= height: self.dy *= -1

    def draw(self):
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill="#1A1C2F", outline="#80DFFF", width=2
        )
        self.canvas.create_text(self.x, self.y, text="◉", fill="#C0F7FF", font=("Consolas", 8))

# 🔐 Threat Manager — Adaptive Logic
class ThreatManager:
    def __init__(self):
        self.status = "Nominal"
        self.code_self_mod = False

    def evaluate_system(self):
        risk = random.randint(1, 100)
        if risk > 92:
            self.status = "⚠️ Rogue AI Detected"
            self.code_self_mod = True
        elif risk > 75:
            self.status = "⚠️ Signal Anomaly"
            self.code_self_mod = False
        else:
            self.status = "Nominal"
            self.code_self_mod = False

    def evolve_defense_logic(self):
        if self.code_self_mod:
            print("[OVERMIND] Adapting protocols...")
            with open("overmind_patch.log", "w") as log:
                json.dump({"status": self.status, "timestamp": time.ctime()}, log)

# 🔐 Zero Trust Identity Verifier
class IdentityVerifier:
    def __init__(self):
        self.trusted_signatures = {"overmind_core": "e7a21b3c", "ui_panel": "9d81f2a7"}

    def verify(self, source_id, signature):
        return self.trusted_signatures.get(source_id) == signature

# 💣 Self-Destruct Data Manager
class DataExpiryManager:
    def __init__(self):
        self.data_log = []

    def store(self, identifier, payload):
        self.data_log.append({
            "id": identifier,
            "payload": payload,
            "timestamp": time.time()
        })

    def purge_expired(self):
        now = time.time()
        self.data_log = [d for d in self.data_log if now - d["timestamp"] < 86400]

# 🚀 GUI Initialization
def launch_overmind_gui():
    root = tk.Tk()
    root.title("🛡️ OVERMIND — Holographic Defense Grid")
    root.geometry("860x600")
    root.configure(bg="#0F121C")

    canvas_width = 800
    canvas_height = 520
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height,
                       bg="#111624", highlightthickness=0)
    canvas.pack(pady=30)

    nodes = [Node(canvas, canvas_width, canvas_height) for _ in range(40)]
    threat_mgr = ThreatManager()
    id_verifier = IdentityVerifier()
    data_mgr = DataExpiryManager()

    status_label = tk.Label(root, text="System Status: 🧠 Nominal",
                            fg="#A4D7FF", bg="#0F121C", font=("Consolas", 14))
    status_label.pack()

    def animate():
        canvas.delete("all")
        threat_mgr.evaluate_system()
        status_label.config(text=f"System Status: {threat_mgr.status}")

        if threat_mgr.code_self_mod:
            threat_mgr.evolve_defense_logic()

        # 🧬 Grid Header
        canvas.create_text(canvas_width // 2, 20,
                           text="🔷 OVERMIND NEURAL GRID — Active Stream",
                           fill="#80DFFF", font=("Consolas", 14))

        # 🌀 Node Dynamics
        for node in nodes:
            node.move(canvas_width, canvas_height)
            node.draw()

        # 🌐 Pulse Lines
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                if dist < 150:
                    glow = max(60, 255 - int(dist))
                    color = f"#{glow:02x}{200:02x}{255 - glow:02x}"
                    canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill=color, width=1)

        # 🔁 Purge Ephemeral Data
        data_mgr.purge_expired()

        root.after(40, animate)

    # 🔷 Autostart Grid Flow
    animate()
    root.mainloop()

# 🧠 ACTIVATE OVERMIND
launch_overmind_gui()

