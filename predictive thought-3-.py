# glyphforge_core.py

# ✅ Auto-install required libraries
required = [
    "numpy", "scipy", "matplotlib", "seaborn",
    "pyttsx3", "torch"
]
import sys, subprocess
for pkg in required:
    try: __import__(pkg)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 🔧 Imports
import numpy as np
import json, random, time
from scipy.fft import rfft, rfftfreq
from datetime import datetime
import torch
import torch.nn as nn

# 🎵 Ghost Resonance Generator
class GhostResonanceSimulator:
    def __init__(self, num_waves=5, length=1000):
        self.num_waves = num_waves
        self.length = length
        self.time = np.linspace(0, 10, length)
        self.pattern = self.generate_resonance()

    def generate_resonance(self):
        resonance = np.zeros(self.length)
        for _ in range(self.num_waves):
            frequency = np.random.uniform(0.5, 5)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.3, 1.0)
            damping = np.exp(-0.1 * self.time)
            wave = amplitude * np.sin(2 * np.pi * frequency * self.time + phase) * damping
            resonance += wave
        return resonance

# 🔤 Symbolic Classifier
class LayeredGlyphClassifier:
    def __init__(self, pattern, sample_rate=100):
        self.pattern = pattern
        self.freqs, self.spectrum = self.analyze(sample_rate)

    def analyze(self, sample_rate):
        spectrum = np.abs(rfft(self.pattern))
        freqs = rfftfreq(len(self.pattern), 1 / sample_rate)
        return freqs, spectrum

    def classify_layers(self):
        bands = {
            "low": (0.1, 1.0),
            "mid": (1.0, 3.0),
            "high": (3.0, 6.0)
        }
        glyph_map = {
            "low": "⟁",
            "mid": "⧉",
            "high": "⨀"
        }
        layers = []
        for band, (low_f, high_f) in bands.items():
            band_mask = (self.freqs >= low_f) & (self.freqs < high_f)
            if np.any(band_mask):
                energy = np.mean(self.spectrum[band_mask])
                if energy > np.mean(self.spectrum):
                    layers.append(glyph_map[band])
        return layers or ["◌"]

# 🧠 Neural GlyphNet (mocked)
class GlyphNet(nn.Module):
    def __init__(self, input_size=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.net(x)

class NeuralGlyphClassifier:
    def __init__(self):
        self.model = GlyphNet()
        self.glyph_map = ["⟁", "⧉", "⨀", "◌"]

    def predict(self, pattern):
        with torch.no_grad():
            x = torch.tensor(pattern[:1000], dtype=torch.float32)
            if x.ndim == 1:
                x = x.view(1, -1)
            out = self.model(x)
            idx = torch.topk(out, 2).indices[0].tolist()
            return [self.glyph_map[i] for i in idx]

# 🧬 Glyph Memory + Trails
class GlyphMemory:
    def __init__(self):
        self.glyph_counts = {}

    def update(self, glyphs):
        for g in glyphs:
            self.glyph_counts[g] = self.glyph_counts.get(g, 0) + 1

    def bias(self):
        total = sum(self.glyph_counts.values())
        return {k: v / total for k, v in self.glyph_counts.items()} if total else {}

class GlyphTrail:
    def __init__(self):
        self.trail = []

    def log(self, node_id, glyphs):
        self.trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "node": node_id,
            "glyphs": glyphs
        })

    def export(self, filename="glyphtrail.sigil"):
        with open(filename, "w") as f:
            json.dump(self.trail, f, indent=2)

# 📜 Ritual Meaning
glyph_meanings = {
    "⟁": "The Foundation Stone: stability beneath turbulence.",
    "⧉": "The Tideform: adaptive resonance and flow.",
    "⨀": "The Phoenix Core: chaotic ignition and rebirth.",
    "◌": "The Hidden Path: glyph of latency and silence."
}

def narrate(glyphs):
    return " | ".join(f"{g}: {glyph_meanings.get(g, 'Unknown resonance')}" for g in glyphs)

def export_mythlog(log_entries, filename="ritual.myth"):
    with open(filename, "w") as f:
        for line in log_entries:
            f.write(line + "\n")

# 🔮 Symbolic Forecaster
class GlyphPredictor:
    def __init__(self):
        self.history = []

    def update(self, glyphs):
        self.history.append(glyphs)
        self.history = self.history[-12:]

    def predict_next(self):
        if len(self.history) < 3:
            return ["◌"]
        last = self.history[-1]
        common = [g for g in last if all(g in h for h in self.history[-3:])]
        return common or ["◌"]

# 📈 Deep Forecast (LSTM Sim)
class GlyphForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 8)
        self.lstm = nn.LSTM(8, 32, batch_first=True)
        self.decoder = nn.Linear(32, 4)
        self.glyph_map = ["⟁", "⧉", "⨀", "◌"]
        self.token_map = {g: i for i, g in enumerate(self.glyph_map)}

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.decoder(out[:, -1])

    def forecast(self, glyph_history):
        tokens = [self.token_map.get(g, 3) for g in glyph_history[-5:]]
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            pred = torch.topk(logits, 2).indices[0].tolist()
            return [self.glyph_map[i] for i in pred]

# glyphforge_swarm.py

# ✅ Auto-install
required = ["random", "json"]
import sys, subprocess
for pkg in required:
    try: __import__(pkg)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 🔧 Imports
import random
from glyphforge_core import glyph_meanings

# 🎭 Glyph Behavior Map
glyph_behavior_map = {
    "⟁": "stabilize",
    "⧉": "synchronize",
    "⨀": "diversify",
    "◌": "idle"
}

# 🐝 Swarm Node Agent
class SwarmNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = "idle"
        self.glyph = "◌"
        self.position = (random.random(), random.random())
        self.history = []

    def update_behavior(self, glyph):
        self.glyph = glyph
        self.state = glyph_behavior_map.get(glyph, "idle")
        self.history.append((glyph, self.state))
        self.history = self.history[-5:]

    def emit_status(self):
        return {
            "id": self.node_id,
            "glyph": self.glyph,
            "state": self.state,
            "pos": self.position
        }

# ⚖️ Reinforcement-Capable Agent Wrapper (for future RL)
class RLAgent:
    def __init__(self, node):
        self.node = node
        self.q_table = {}
        self.actions = ["stabilize", "synchronize", "diversify", "idle"]

    def observe_state(self):
        return self.node.glyph

    def choose_action(self, epsilon=0.2):
        state = self.observe_state()
        if random.random() < epsilon or state not in self.q_table:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q(self, reward, alpha=0.1, gamma=0.9):
        state = self.observe_state()
        action = self.node.state
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        old_value = self.q_table[state][action]
        new_value = old_value + alpha * (reward - old_value)
        self.q_table[state][action] = new_value

# 🔁 Ritual Transition System
class RitualTransition:
    def __init__(self):
        self.last_glyphs = set()

    def detect_transitions(self, new_glyphs):
        new = set(new_glyphs)
        changes = new - self.last_glyphs
        arcs = []
        for g in changes:
            if "⨀" in self.last_glyphs and g == "⧉":
                arcs.append("↷ Ritual Arc: Chaos reborn as Flux (⨀→⧉)")
        self.last_glyphs = new
        return arcs

# 🌐 Swarm Network Fabric
class SwarmNetwork:
    def __init__(self):
        self.nodes = []
        self.agents = []

    def sync_nodes(self, count=10):
        self.nodes = [SwarmNode(f"Node-{i}") for i in range(count)]
        self.agents = [RLAgent(n) for n in self.nodes]

    def broadcast_glyph(self, glyphs):
        for node, agent in zip(self.nodes, self.agents):
            for g in glyphs:
                node.update_behavior(g)
                # Example reward: prefer synchronized state
                reward = 1.0 if node.state == "synchronize" else -0.2
                agent.update_q(reward)

    def observe(self):
        return [n.emit_status() for n in self.nodes]

# glyphforge_dashboard.py

# ✅ Auto-install
required = [
    "numpy", "matplotlib", "seaborn", "pyttsx3", "scipy", "torch"
]
import sys, subprocess
for pkg in required:
    try: __import__(pkg)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 🔧 Imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pyttsx3

from glyphforge_core import (
    GhostResonanceSimulator, LayeredGlyphClassifier,
    NeuralGlyphClassifier, GlyphForecaster,
    GlyphMemory, GlyphTrail, GlyphPredictor,
    narrate, export_mythlog
)
from glyphforge_swarm import SwarmNetwork, RitualTransition

# 🎨 Glyph Colors
glyph_colors = {
    "⟁": "#4CAF50", "⧉": "#2196F3",
    "⨀": "#F44336", "◌": "#9E9E9E"
}

# 🎙️ Voice Engine
enable_voice = True
voice_engine = pyttsx3.init() if enable_voice else None

# 🧠 Engine Modes
use_neural = True       # toggle for neural glyph predictor
deep_forecast = True    # toggle for LSTM glyph prediction

# 🧬 Modules
sim = GhostResonanceSimulator()
trail = GlyphTrail()
memory = GlyphMemory()
predictor = GlyphPredictor()
forecaster = GlyphForecaster()
swarm = SwarmNetwork()
swarm.sync_nodes(10)
rituals = RitualTransition()
narration_log = []
glyph_history = []

# 📊 Layout
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 3)
ax_wave = fig.add_subplot(gs[0, :])
ax_trail = fig.add_subplot(gs[1, 0])
ax_constellation = fig.add_subplot(gs[1, 1])
ax_matrix = fig.add_subplot(gs[1, 2])
ax_feed = fig.add_subplot(gs[2, :])

# 🔁 Dashboard Loop
def update_dashboard(frame):
    global glyph_history

    pattern = sim.generate_resonance()

    if use_neural:
        neural = NeuralGlyphClassifier()
        glyphs = neural.predict(pattern)
    else:
        glyphs = LayeredGlyphClassifier(pattern).classify_layers()

    trail.log("core", glyphs)
    memory.update(glyphs)
    predictor.update(glyphs)
    symbolic_forecast = predictor.predict_next()
    neural_forecast = forecaster.forecast(glyphs) if deep_forecast else ["◌"]
    arcs = rituals.detect_transitions(glyphs)

    narration = narrate(glyphs)
    forecast_str = " ".join(symbolic_forecast)
    neural_str = " ".join(neural_forecast)
    narration_log.append(narration)
    narration_log[:] = narration_log[-5:]

    # 🔈 Voice
    if enable_voice and voice_engine:
        voice_engine.say(f"Glyphs: {' '.join(glyphs)}. Forecast: {forecast_str}")
        voice_engine.runAndWait()

    # 🌀 Waveform
    ax_wave.clear()
    ax_wave.plot(sim.time, pattern, color="cyan")
    ax_wave.set_title("Ghost Resonance")

    # ⛩ Trail
    glyph_history.append("".join(glyphs))
    glyph_history = glyph_history[-20:]
    ax_trail.clear()
    ax_trail.text(0.1, 0.5, " ".join(glyph_history), fontsize=18, fontfamily="monospace")
    ax_trail.set_title("Glyph Trail")
    ax_trail.axis("off")

    # 🌌 Constellation
    ax_constellation.clear()
    swarm.broadcast_glyph(glyphs)
    for node in swarm.observe():
        x, y = node["pos"]
        g = node["glyph"]
        ax_constellation.text(x, y, g, fontsize=14, color=glyph_colors[g], transform=ax_constellation.transAxes,
                              ha='center', va='center')
    ax_constellation.set_title("Swarm Glyph Constellation")
    ax_constellation.axis("off")

    # 🔥 Matrix
    ax_matrix.clear()
    counts = memory.glyph_counts
    data = [[counts.get(g, 0)] for g in ["⟁", "⧉", "⨀", "◌"]]
    sns.heatmap(data, annot=True, cmap="magma", yticklabels=["⟁","⧉","⨀","◌"], cbar=False, ax=ax_matrix)
    ax_matrix.set_title("Sigil Memory")

    # 📜 Feed
    ax_feed.clear()
    ax_feed.text(0.05, 0.75, narration, fontsize=11)
    ax_feed.text(0.05, 0.5, "\n".join(arcs) or "—", fontsize=10, color="purple")
    ax_feed.text(0.05, 0.3, f"🜄 Symbolic Forecast: {forecast_str}", fontsize=10, color="gray")
    ax_feed.text(0.05, 0.15, f"🧠 Neural Forecast: {neural_str}", fontsize=10, color="teal")
    ax_feed.axis("off")
    ax_feed.set_title("Mythic Narration")

# 🚀 Launch
ani = animation.FuncAnimation(fig, update_dashboard, interval=1500)
plt.tight_layout()
plt.show()

# 💾 Export
export_mythlog(narration_log)
trail.export("glyphtrail.sigil")

# ✅ Auto-install essentials
required = ["pyttsx3"]
import sys, subprocess
for pkg in required:
    try: __import__(pkg)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import argparse, os, pyttsx3

def speak_intro():
    engine = pyttsx3.init()
    engine.say("Welcome to Glyphforge Neural Interface. The forge is awake.")
    engine.runAndWait()

def launch_dashboard(mute=False, neural=False, forecast=False):
    os.environ["GLYPHFORGE_MUTE"] = "1" if mute else "0"
    os.environ["GLYPHFORGE_NEURAL"] = "1" if neural else "0"
    os.environ["GLYPHFORGE_DEEP"] = "1" if forecast else "0"
    os.system(f"{sys.executable} glyphforge_dashboard.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Glyphforge Ritual Interface")
    parser.add_argument("--mute", action="store_true", help="Disable voice narration")
    parser.add_argument("--neural", action="store_true", help="Use neural glyph cognition")
    parser.add_argument("--deep-forecast", action="store_true", help="Use deep glyph forecast (LSTM)")
    args = parser.parse_args()

    if not args.mute:
        speak_intro()

    print("\n🌌 GLYPHFORGE INITIATED")
    print("🜁 Voice:", "Muted" if args.mute else "Active")
    print("🧠 Neural Cognition:", "Enabled" if args.neural else "Symbolic Heuristics")
    print("🔮 Deep Forecast:", "Enabled" if args.deep_forecast else "Symbolic Prediction")
    print("🜂 Rituals:", "Real-time")
    print("🜄 Glyph memory:", "glyphtrail.sigil\n")

    launch_dashboard(mute=args.mute, neural=args.neural, forecast=args.deep_forecast)

python glyphforge_launcher.py --neural --deep-forecast
