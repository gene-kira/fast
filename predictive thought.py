# glyphforge_core.py

import numpy as np
import json, random, time
from scipy.fft import rfft, rfftfreq
from datetime import datetime
import os

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

# 🔤 Layered Glyph Classifier
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
        return layers or ["◌"]  # Empty glyph

# 🪶 Glyph Memory + Trail
class GlyphMemory:
    def __init__(self):
        self.glyph_counts = {}

    def update(self, glyphs):
        for glyph in glyphs:
            self.glyph_counts[glyph] = self.glyph_counts.get(glyph, 0) + 1

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

# 📜 Narrative Generator
glyph_meanings = {
    "⟁": "The Foundation Stone: stability beneath turbulence.",
    "⧉": "The Tideform: adaptive resonance and flow.",
    "⨀": "The Phoenix Core: chaotic ignition and rebirth.",
    "◌": "The Hidden Path: glyph of latency and silence."
}

def narrate(glyphs):
    return " | ".join(f"{g}: {glyph_meanings.get(g, 'Unknown resonance')}" for g in glyphs)

# 📦 Ritual Log Exporter
def export_mythlog(log_entries, filename="ritual.myth"):
    with open(filename, "w") as f:
        for entry in log_entries:
            f.write(entry + "\n")

# glyphforge_swarm.py

import random
from glyphforge_core import glyph_meanings

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
        self.position = (random.random(), random.random())  # Normalized 0–1 space

    def update_behavior(self, glyph):
        self.glyph = glyph
        self.state = glyph_behavior_map.get(glyph, "idle")

    def emit_status(self):
        return {
            "id": self.node_id,
            "glyph": self.glyph,
            "state": self.state,
            "pos": self.position
        }

# 🔁 Ritual Transition Arcs
class RitualTransition:
    def __init__(self):
        self.last_glyphs = set()

    def detect_transitions(self, new_glyphs):
        new = set(new_glyphs)
        changes = new - self.last_glyphs
        arcs = []
        for g in changes:
            if "⨀" in self.last_glyphs and g == "⧉":
                arcs.append(f"↷ Ritual Arc: Chaos reborn as Flux (⨀→⧉)")
        self.last_glyphs = new
        return arcs

# 🌐 (Optional) Network Sync Stub
class SwarmNetwork:
    def __init__(self):
        self.nodes = []

    def broadcast_glyph(self, glyphs):
        for node in self.nodes:
            for g in glyphs:
                node.update_behavior(g)

    def sync_nodes(self, count=10):
        self.nodes = [SwarmNode(f"Node-{i}") for i in range(count)]

    def observe(self):
        return [n.emit_status() for n in self.nodes]

# glyphforge_dashboard.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from glyphforge_core import GhostResonanceSimulator, LayeredGlyphClassifier, GlyphMemory, GlyphTrail, narrate, export_mythlog
from glyphforge_swarm import SwarmNetwork, RitualTransition
import pyttsx3

# 🎨 Theme Palette
glyph_colors = {
    "⟁": "#4CAF50",  # Grounded
    "⧉": "#2196F3",  # Flow
    "⨀": "#F44336",  # Chaos
    "◌": "#9E9E9E"   # Latent
}

# 🎙️ Voice Toggle
enable_voice = True
voice_engine = pyttsx3.init() if enable_voice else None

# 🧠 Init modules
sim = GhostResonanceSimulator()
memory = GlyphMemory()
trail = GlyphTrail()
swarm = SwarmNetwork()
swarm.sync_nodes(10)
rituals = RitualTransition()
narration_log = []

# 📊 Matplotlib setup
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 3)

ax_wave = fig.add_subplot(gs[0, :])
ax_trail = fig.add_subplot(gs[1, 0])
ax_constellation = fig.add_subplot(gs[1, 1])
ax_matrix = fig.add_subplot(gs[1, 2])
ax_feed = fig.add_subplot(gs[2, :])

glyph_history = []

# 🔁 Dashboard Update Function
def update_dashboard(frame):
    global glyph_history

    pattern = sim.generate_resonance()
    classifier = LayeredGlyphClassifier(pattern)
    glyphs = classifier.classify_layers()
    trail.log("core", glyphs)
    memory.update(glyphs)
    arcs = rituals.detect_transitions(glyphs)

    # Voice narration
    if enable_voice and voice_engine:
        voice_engine.say(narrate(glyphs))
        voice_engine.runAndWait()

    # 🎵 Visualization Panels
    ax_wave.clear()
    ax_wave.plot(sim.time, pattern, color="cyan")
    ax_wave.set_title("⧉ Ghost Resonance Pattern")

    # 🌀 Trail
    glyph_history.append("".join(glyphs))
    glyph_history = glyph_history[-20:]
    ax_trail.clear()
    ax_trail.text(0.1, 0.5, " ".join(glyph_history), fontsize=20, fontfamily="monospace", color="#222")
    ax_trail.set_title("Glyph Trail")
    ax_trail.axis("off")

    # 🌌 Constellation
    ax_constellation.clear()
    swarm.broadcast_glyph(glyphs)
    for node in swarm.observe():
        x, y = node["pos"]
        glyph = node["glyph"]
        ax_constellation.text(x, y, glyph, fontsize=14, color=glyph_colors[glyph], transform=ax_constellation.transAxes,
                              ha='center', va='center')
    ax_constellation.set_title("Swarm Glyph Constellation")
    ax_constellation.axis("off")

    # 🔥 Sigil Matrix
    ax_matrix.clear()
    counts = memory.glyph_counts
    data = [[counts.get(g, 0)] for g in ["⟁", "⧉", "⨀", "◌"]]
    sns.heatmap(data, annot=True, cmap="magma", yticklabels=["⟁","⧉","⨀","◌"], cbar=False, ax=ax_matrix)
    ax_matrix.set_title("Sigil Memory Matrix")

    # 📖 Narrative Feed
    arc_lines = "\n".join(arcs) if arcs else "Awaiting ritual transitions..."
    narration = narrate(glyphs)
    narration_log.append(narration)
    narration_log[:] = narration_log[-3:]
    ax_feed.clear()
    ax_feed.text(0.05, 0.7, narration, fontsize=12)
    ax_feed.text(0.05, 0.4, arc_lines, fontsize=10, color="purple")
    ax_feed.axis("off")
    ax_feed.set_title("Mythic Narration")

# 🚀 Animate
ani = animation.FuncAnimation(fig, update_dashboard, interval=1500)
plt.tight_layout()
plt.show()

# 📤 Optional Export
export_mythlog(narration_log)
trail.export("glyphtrail.sigil")

