# 🔧 Autoloader: Install and import required libraries
import subprocess
import sys

def autoload(package, import_as=None):
    try:
        globals()[import_as or package] = __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[import_as or package] = __import__(package)

# 📦 Required packages
autoload("pygame")
autoload("pyttsx3")

# ✅ Explicit imports
import tkinter as tk
import threading
import time
import math
import random
import json

pygame.mixer.init()
engine = pyttsx3.init()
voice_alert_active = False

# 🧬 Agent Registry
agent_registry = {
    "Seeker_453": {"role": "Seeker", "trust": 0.67, "token": "token_453"},
    "Sentinel_007": {"role": "Sentinel", "trust": 0.91, "token": "token_007"},
    "Archivist_314": {"role": "Archivist", "trust": 0.88, "token": "token_314"}
}

# 📜 Glyph Lineage Memory
glyph_lineage = {
    "🧿_Glyph_001": {"ancestry": ["🌀_Initiation"], "verdicts": ["Canonized"], "traits": ["Resilient"]},
    "🧿_Glyph_002": {"ancestry": ["🧿_Glyph_001"], "verdicts": ["Remixed"], "traits": []}
}

# 🗃️ Memory Graph Logger
def log_event(event_type, agent_id, glyph_id, verdict):
    entry = {
        "event": event_type,
        "agent": agent_id,
        "glyph": glyph_id,
        "verdict": verdict,
        "role": agent_registry[agent_id]["role"],
        "trust": agent_registry[agent_id]["trust"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("memory_graph.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

# 🔊 Tone Generator
def play_tone(freq=440, duration=500, volume=0.5):
    sample_rate = 44100
    n_samples = int(sample_rate * duration / 1000)
    wave = (volume * 32767 * pygame.sndarray.array([
        [int(32767 * pygame.math.sin(2.0 * pygame.math.pi * freq * x / sample_rate))] * 2
        for x in range(n_samples)
    ])).astype("int16")
    buffer = pygame.sndarray.make_sound(wave)
    buffer.play()

# 🗣️ Voice Alert
def speak_phrase(phrase):
    global voice_alert_active
    if voice_alert_active:
        engine.say(phrase)
        engine.runAndWait()

# 🔇 ACK Button Handler
def silence_alert(agent_id):
    global voice_alert_active
    if can_ack(agent_id):
        voice_alert_active = False
        log_event("VoiceAlertAck", agent_id, "VoiceAlert", "Silenced")
        print(f"🔇 Voice alert silenced by {agent_id}")
    else:
        print(f"🚫 {agent_id} not authorized to ACK.")

def can_ack(agent_id):
    agent = agent_registry.get(agent_id)
    return agent and agent["role"] in ["Sentinel", "Archivist"] and agent["trust"] > 0.7

# 💥 Self-Destruct Protocol
def trigger_self_destruct(agent_id):
    if can_self_destruct(agent_id):
        print(f"💥 Self-destruct triggered by {agent_id}")
        play_tone(90, 1000, 1.0)
        speak_phrase(f"Emergency protocol activated. Data purge initiated by {agent_id}.")
        log_event("SelfDestruct", agent_id, "🧿_StolenNode_314", "Purged")
    else:
        print(f"🚫 {agent_id} not authorized for self-destruct.")

def can_self_destruct(agent_id):
    agent = agent_registry.get(agent_id)
    return agent and agent["role"] in ["Sentinel", "Archivist"] and agent["trust"] > 0.85

# 🧿 Glyph Node
class GlyphNode:
    def __init__(self, canvas, glyph_id, trust, ancestry):
        self.canvas = canvas
        self.glyph_id = glyph_id
        self.trust = trust
        self.angle = random.uniform(0, 360)
        self.orbit_radius = 80 + trust * 100
        self.orbit_speed = 0.5 + (1.0 - trust)
        self.color = self.trust_to_color(trust)
        self.x, self.y = 0, 0
        self.id = canvas.create_oval(0, 0, 0, 0, fill=self.color, outline="white", width=2)
        self.trail = None
        self.ancestry = ancestry
        self.traits = glyph_lineage.get(glyph_id, {}).get("traits", [])

    def trust_to_color(self, trust):
        if trust > 0.8:
            return "#00FFAA"
        elif trust > 0.5:
            return "#88FF00"
        else:
            return "#FF4444"

    def update(self, center_x, center_y):
        self.angle += self.orbit_speed
        self.x = center_x + self.orbit_radius * math.cos(math.radians(self.angle))
        self.y = center_y + self.orbit_radius * math.sin(math.radians(self.angle))
        r = int(10 + self.trust * 20)
        self.canvas.coords(self.id, self.x - r, self.y - r, self.x + r, self.y + r)
        if self.ancestry and not self.trail:
            self.trail = self.canvas.create_line(center_x, center_y, self.x, self.y, fill="#8888FF", dash=(2, 2))

# 🌌 Dashboard Manager
class ConstellationDashboard:
    def __init__(self, canvas):
        self.canvas = canvas
        self.glyphs = {}

    def spawn_glyph(self, glyph_id):
        trust = random.uniform(0.3, 1.0)
        ancestry = glyph_lineage.get(glyph_id, {}).get("ancestry", [])
        glyph = GlyphNode(self.canvas, glyph_id, trust, ancestry)
        self.glyphs[glyph_id] = glyph

    def update_all(self):
        for glyph in self.glyphs.values():
            glyph.update(200, 200)

    def annotate_glyph(self, agent_id, glyph_id, verdict):
        if glyph_id in self.glyphs:
            glyph_lineage[glyph_id]["verdicts"].append(verdict)
            glyph_lineage[glyph_id]["traits"].append(verdict)
            log_event("GlyphAnnotation", agent_id, glyph_id, verdict)
            print(f"📝 {agent_id} annotated {glyph_id} with '{verdict}'")

# 🖥️ GUI Setup
def launch_gui():
    root = tk.Tk()
    root.title("🌌 Constellation Dashboard")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    canvas.create_oval(190, 190, 210, 210, fill="#00FFFF", outline="white", width=2)
    canvas.create_text(200, 180, text="🧠 Ritual Core", fill="white", font=("Helvetica", 10))

    dashboard = ConstellationDashboard(canvas)

    def spawn_event():
        glyph_id = random.choice(list(glyph_lineage.keys()))
        dashboard.spawn_glyph(glyph_id)

    def annotate_event():
        agent_id = random.choice(list(agent_registry.keys()))
        glyph_id = random.choice(list(dashboard.glyphs.keys()))
        verdict = random.choice(["Canonized", "Remixed", "Purged"])
        dashboard.annotate_glyph(agent_id, glyph_id, verdict)

    def ack_event():
        agent_id = random.choice(list(agent_registry.keys()))
        silence_alert(agent_id)

    def destruct_event():
        agent_id = random.choice(list(agent_registry.keys()))
        trigger_self_destruct(agent_id)

    tk.Button(root, text="Emit Glyph", command=spawn_event, font=("Helvetica", 10)).pack(pady=2)
    tk.Button(root, text="Seal Verdict", command=annotate_event, font=("Helvetica", 10)).pack(pady=2)
    tk.Button(root, text="🔇 ACK Alert", command=ack_event, font=("Helvetica", 10)).pack(pady=2)
    tk.Button(root, text="💥 Self-Destruct", command=destruct_event, font=("Helvetica", 10), bg="#FF4444", fg="white").pack(pady=2)

    def animate_loop():
        global voice_alert_active
        voice_alert_active = True
        speak_phrase("Constellation dashboard initialized.")
        while True:
            dashboard.update_all()
            time.sleep(0.05)

    threading.Thread(target=animate_loop, daemon=True).start()
    root.mainloop()

# 🚀 Launch Ritual Dashboard
if __name__ == "__main__":
    launch_gui()

