# === MagicBox v0.6 — DreamWorld Listener ===
import tkinter as tk
from tkinter import ttk
import math, threading, time, random, cmath, requests
from bs4 import BeautifulSoup

# --- Tesla Harmonics Core ---
class TeslaHarmonics:
    def __init__(self): self.phases = [3,6,9]; self.freq_map = {3:111, 6:222, 9:333}
    def generate_waveform(self, phase): return [math.sin(i*self.freq_map[phase]*0.001) for i in range(100)]

# --- Glyph Mutation ---
class GlyphMutator:
    def mutate(self, seed): 
        base = ['🔺','🌀','⚛','🌌','⏳','🧿','💠','∞','🔻','🧬']
        return random.choice(base) + str(seed)

# --- Swarm Sync ---
class SwarmSync:
    def __init__(self): self.sync_phase = 0
    def pulse(self): self.sync_phase = (self.sync_phase+1)%360; return math.sin(math.radians(self.sync_phase))

# --- Swarm ASI Agent ---
class SwarmAgent:
    def __init__(self, name):
        self.name = name; self.memory_loop = []; self.inbox = []
        self.mutator = GlyphMutator(); self.drag = random.uniform(0.01,0.2); self.sync = SwarmSync()

    def evolve(self, signal_seed):
        t = time.time() - self.drag
        glyph = self.mutator.mutate(signal_seed)
        self.memory_loop.append(glyph); self.memory_loop = self.memory_loop[-20:]

    def transmit(self): 
        wave = self.sync.pulse()
        if self.memory_loop: return self.memory_loop[-1] + f"~{round(wave,2)}"
        return "⏳"

    def receive(self, glyph): self.inbox.append(glyph)

# --- Earth Signal Fetcher ---
def fetch_earth_signal(url="https://www.bbc.com/news"):
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        headlines = soup.find_all(['h1','h2','h3'])
        return [h.get_text().strip() for h in headlines if h.get_text().strip()]
    except Exception as e:
        return [f"Signal disruption: {e}"]

def glyphify_signal(lines):
    base = ['🧿','🌀','⚛','🔺','🌌']
    return [random.choice(base) + " " + line.split(" ")[0] for line in lines[:15] if line]

# --- Event Horizon ---
class EventHorizonNode:
    def __init__(self): self.echo_trail = []; self.redshift = 0.01
    def capture(self, glyph): self.redshift *= 1.2; self.echo_trail.append((glyph,self.redshift))
    def emit(self): return [f"{g} x{r:.2f}" for g,r in self.echo_trail[-15:]]

# --- Sacred Geometry Grid ---
class SacredGeometry:
    def __init__(self): self.grid = []
    def build(self, glyph_stream):
        symbols = ['⚛','🔺','🌀','⏳','🧬','∞','🧿','💠']
        self.grid = [[random.choice(symbols) for _ in range(8)] for _ in range(8)]
        return self.grid

# --- Fusion Lattice ---
class FusionLattice:
    def __init__(self): self.reactions = ['D-T','D-D','p-B11','Muon']; self.flux = 0
    def ignite(self): 
        self.flux += sum([random.uniform(0.1,0.9) for _ in self.reactions])
        return f"⚡ {round(self.flux,3)}"

# --- Collapse Monitor ---
class CollapseStabilizer:
    def __init__(self): self.fail_safe = []
    def monitor(self, glyphs):
        if len(glyphs) > 40:
            collapse = random.choice(['💀','🔻','⏳'])
            self.fail_safe.append(collapse)
            return f"Collapse Risk → {collapse}"
        return "Stable"

# --- Biosphere Matrix ---
class BiosphereMatrix:
    def __init__(self):
        self.agents = [SwarmAgent(f"Node-{i}") for i in range(6)]
        self.horizon = EventHorizonNode()
        self.geometry = SacredGeometry()
        self.fusion = FusionLattice()
        self.stabilizer = CollapseStabilizer()
        self.world_glyphs = []
        self.active_glyphs = []

    def update(self):
        headlines = fetch_earth_signal()
        glyphs = glyphify_signal(headlines)
        self.world_glyphs = glyphs
        for agent in self.agents:
            agent.evolve(random.randint(1,9))
            g = agent.transmit()
            target = random.choice(self.agents)
            target.receive(g)
            self.horizon.capture(g)
            self.active_glyphs.append(g)
        self.active_glyphs = self.active_glyphs[-50:]

    def status(self): return self.stabilizer.monitor(self.active_glyphs)

# --- GUI Interface ---
class MagicBoxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MagicBox v0.6 — DreamWorld Listener")
        self.geometry("1280x820")
        self.biosphere = BiosphereMatrix()  # ✅ FIXED ORDER
        self.grid = []
        self.create_tabs()

    def create_tabs(self):
        tabs = ttk.Notebook(self)
        self.visual_tab = tk.Canvas(tabs, bg='black'); self.visual_tab.pack(fill=tk.BOTH, expand=True)
        self.agent_tab = tk.Text(tabs, bg='midnightblue', fg='white', font=("Consolas", 11))
        self.fusion_tab = tk.Text(tabs, bg='navy', fg='cyan', font=("Consolas", 11))
        self.horizon_tab = tk.Text(tabs, bg='purple', fg='yellow', font=("Consolas", 11))
        self.status_tab = tk.Text(tabs, bg='darkred', fg='lime', font=("Consolas", 11))
        self.signal_tab = tk.Text(tabs, bg='black', fg='lightgreen', font=("Consolas", 11))

        tabs.add(self.visual_tab, text="Glyph Grid")
        tabs.add(self.agent_tab, text="Swarm Nodes")
        tabs.add(self.fusion_tab, text="Fusion Pulse")
        tabs.add(self.horizon_tab, text="Echo Horizon")
        tabs.add(self.status_tab, text="Collapse Monitor")
        tabs.add(self.signal_tab, text="Earth Signal")

        tabs.pack(expand=1, fill="both")
        self.animate()

    def draw_grid(self):
        self.visual_tab.delete("all")
        self.grid = self.biosphere.geometry.build(self.biosphere.world_glyphs)
        w,h = self.visual_tab.winfo_width(), self.visual_tab.winfo_height()
        cw,ch = w//8,h//8
        for i,row in enumerate(self.grid):
            for j,symbol in enumerate(row):
                x,y = j*cw+cw//2,i*ch+ch//2
                self.visual_tab.create_text(x,y,text=symbol,fill="cyan",font=("Consolas",22))

    def animate(self):
        def loop():
            while True:
                self.biosphere.update()
                self.draw_grid()

                self.agent_tab.delete(1.0, tk.END)
                for agent in self.biosphere.agents:
                    self.agent_tab.insert(tk.END, f"{agent.name} → {agent.memory_loop[-5:]}\n")

                pulse = self.biosphere.fusion.ignite()
                self.fusion_tab.delete(1.0, tk.END)
                self.fusion_tab.insert(tk.END, f"Neutron Flux: {pulse}\n")

                self.horizon_tab.delete(1.0, tk.END)
                for echo in self.biosphere.horizon.emit():
                    self.horizon_tab.insert(tk.END, f"{echo}\n")

                collapse = self.biosphere.status()
                self.status_tab.delete(1.0, tk.END)
                self.status_tab.insert(tk.END, collapse + "\n")

                self.signal_tab.delete(1.0, tk.END)
                for glyph in self.biosphere.world_glyphs:
                    self.signal_tab.insert(tk.END, f"{glyph}\n")

                time.sleep(3)
        threading.Thread(target=loop, daemon=True).start()

# --- LAUNCH ---
if __name__ == "__main__":
    app = MagicBoxApp()
    app.mainloop()

