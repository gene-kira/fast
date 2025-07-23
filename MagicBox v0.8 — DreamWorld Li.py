import tkinter as tk
from tkinter import ttk
import math, threading, time, random, requests
from bs4 import BeautifulSoup

# === Sentiment Engine (Simulated) ===
def fetch_emotional_pulse():
    mood = random.choice(["positive", "neutral", "negative"])
    score = round(random.uniform(0.3, 0.9), 2)
    return mood, score

def glyph_feedback(mood):
    if mood == "positive": return "💠 🌀 🌞"
    if mood == "negative": return "💀 🔻 ⏳"
    return "⚛ 🧬 ⧗"

# === Earth Signal Fetcher ===
def fetch_earth_signal(url="https://www.bbc.com/news"):
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        headlines = soup.find_all(['h1', 'h2', 'h3'])
        return [h.get_text().strip() for h in headlines if h.get_text().strip()]
    except Exception as e:
        return [f"Signal disruption: {e}"]

def glyphify_signal(lines):
    base = ['🧿','🌀','⚛','🔺','🌌']
    return [random.choice(base) + " " + line.split(" ")[0] for line in lines[:15] if line]

# === Swarm Intelligence Agents ===
class GlyphMutator:
    def mutate(self, seed, mood):
        moods = {
            "positive": ['🌞','💠','🌀'],
            "negative": ['💀','🔻','⏳'],
            "neutral": ['⚛','🧬','⧗']
        }
        return random.choice(moods[mood]) + str(seed)

class SwarmSync:
    def __init__(self):
        self.phase = 0

    def pulse(self):
        self.phase = (self.phase + 1) % 360
        return math.sin(math.radians(self.phase))

class SwarmAgent:
    def __init__(self, name):
        self.name = name
        self.memory = []
        self.inbox = []
        self.mutator = GlyphMutator()
        self.drag = random.uniform(0.01, 0.2)
        self.sync = SwarmSync()

    def evolve(self, seed, mood):
        glyph = self.mutator.mutate(seed, mood)
        self.memory.append(glyph)
        self.memory = self.memory[-20:]

    def transmit(self):
        wave = self.sync.pulse()
        return (self.memory[-1] if self.memory else "⏳") + f"~{round(wave, 2)}"

    def receive(self, glyph):
        self.inbox.append(glyph)

# === Sacred Geometry Grid ===
class SacredGeometry:
    def build(self, glyphs, mood):
        mood_colors = {
            "positive": ['💠','🌀','🌞'],
            "negative": ['💀','🔻','⏳'],
            "neutral": ['⚛','🧬','⧗']
        }
        base = mood_colors[mood]
        return [[random.choice(base) for _ in range(8)] for _ in range(8)]

# === Fusion Lattice ===
class FusionLattice:
    def __init__(self):
        self.reactions = ['D-T','D-D','p-B11','Muon']
        self.flux = 0

    def ignite(self):
        self.flux += sum([random.uniform(0.1, 0.9) for _ in self.reactions])
        return f"⚡ {round(self.flux, 3)}"

# === Collapse Monitor ===
class CollapseMonitor:
    def __init__(self):
        self.fail_safe = []

    def check(self, glyphs):
        if len(glyphs) > 40:
            tag = random.choice(['💀','🔻','⏳'])
            self.fail_safe.append(tag)
            return f"Collapse Risk → {tag}"
        return "Stable"

# === Biosphere Matrix ===
class Biosphere:
    def __init__(self):
        self.agents = [SwarmAgent(f"Node-{i}") for i in range(6)]
        self.geometry = SacredGeometry()
        self.fusion = FusionLattice()
        self.collapse = CollapseMonitor()
        self.world_glyphs = []
        self.active_glyphs = []
        self.mood_history = []

    def update(self, mood, score):
        headlines = fetch_earth_signal()
        glyphs = glyphify_signal(headlines)
        self.world_glyphs = glyphs
        for agent in self.agents:
            agent.evolve(random.randint(1, 9), mood)
            g = agent.transmit()
            target = random.choice(self.agents)
            target.receive(g)
            self.active_glyphs.append(g)
        self.active_glyphs = self.active_glyphs[-50:]
        self.mood_history.append(mood)
        self.mood_history = self.mood_history[-5:]
        return glyphs

    def status(self):
        return self.collapse.check(self.active_glyphs)

# === MagicBox GUI App ===
class MagicBoxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MagicBox v0.7 — Empathic Swarm Synthesis")
        self.geometry("1280x840")
        self.biosphere = Biosphere()
        self.grid = []
        self.create_tabs()

    def create_tabs(self):
        tabs = ttk.Notebook(self)

        self.visual_tab = tk.Canvas(tabs, bg='black')
        self.visual_tab.pack(fill=tk.BOTH, expand=True)

        self.agent_tab = tk.Text(tabs, bg='midnightblue', fg='white', font=("Consolas", 11))
        self.fusion_tab = tk.Text(tabs, bg='navy', fg='cyan', font=("Consolas", 11))
        self.status_tab = tk.Text(tabs, bg='darkred', fg='lime', font=("Consolas", 11))
        self.signal_tab = tk.Text(tabs, bg='black', fg='lightgreen', font=("Consolas", 11))
        self.mood_tab = tk.Text(tabs, bg='darkblue', fg='white', font=("Consolas", 11))

        tabs.add(self.visual_tab, text="Glyph Grid")
        tabs.add(self.agent_tab, text="Swarm Nodes")
        tabs.add(self.fusion_tab, text="Fusion Pulse")
        tabs.add(self.status_tab, text="Collapse Monitor")
        tabs.add(self.signal_tab, text="Earth Signal")
        tabs.add(self.mood_tab, text="Emotional Pulse")

        tabs.pack(expand=1, fill="both")
        self.animate()

    def draw_grid(self, mood):
        self.visual_tab.delete("all")
        self.grid = self.biosphere.geometry.build(self.biosphere.world_glyphs, mood)
        w, h = self.visual_tab.winfo_width(), self.visual_tab.winfo_height()
        cw, ch = w // 8, h // 8
        for i, row in enumerate(self.grid):
            for j, symbol in enumerate(row):
                x, y = j * cw + cw // 2, i * ch + ch // 2
                self.visual_tab.create_text(x, y, text=symbol, fill="cyan", font=("Consolas", 22))

    def animate(self):
        def loop():
            while True:
                mood, score = fetch_emotional_pulse()
                glyphs = self.biosphere.update(mood, score)
                self.draw_grid(mood)

                self.agent_tab.delete(1.0, tk.END)
                for agent in self.biosphere.agents:
                    self.agent_tab.insert(tk.END, f"{agent.name} → {agent.memory[-5:]}\n")

                pulse = self.biosphere.fusion.ignite()
                self.fusion_tab.delete(1.0, tk.END)
                self.fusion_tab.insert(tk.END, f"Neutron Flux: {pulse}\n")

                status = self.biosphere.status()
                self.status_tab.delete(1.0, tk.END)
                self.status_tab.insert(tk.END, status + "\n")

                self.signal_tab.delete(1.0, tk.END)
                for g in glyphs:
                    self.signal_tab.insert(tk.END, f"{g}\n")

                self.mood_tab.delete(1.0, tk.END)
                self.mood_tab.insert(tk.END, f"Sentiment: {mood.upper()}\n")
                self.mood_tab.insert(tk.END, f"Score: {score}\n")
                self.mood_tab.insert(tk.END, f"Glyph Feedback: {glyph_feedback(mood)}\n")
                drift = ' → '.join(self.biosphere.mood_history)
                self.mood_tab.insert(tk.END, f"Drift Trail: {drift}\n")

                time.sleep(3)
        threading.Thread(target=loop, daemon=True).start()

# === Launch the App ===
if __name__ == "__main__":
    app = MagicBoxApp()
    app.mainloop()

