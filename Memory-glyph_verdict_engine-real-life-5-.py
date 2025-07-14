# 🔮 Autoloader: Ensure required libraries are installed
import subprocess
import sys

required_libraries = ["pyttsx3", "matplotlib", "seaborn"]

def ensure_packages(packages):
    for package in packages:
        try:
            if package == "pyttsx3":
                import pyttsx3
            elif package == "matplotlib":
                import matplotlib
            elif package == "seaborn":
                import seaborn
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure_packages(required_libraries)

# 🌐 Imports (after ensuring they exist)
import tkinter as tk
from tkinter import messagebox
import pyttsx3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import random
import math

# 🗣 Voice setup
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# 🌌 Glyph and Verdict Logic
def scan_memory():
    return [{"glyph": random.choice(["⟁", "⧉", "⨀", "◌"]), "trust": random.uniform(0.3, 0.9)} for _ in range(10)]

def initialize_trust():
    return [random.uniform(0.4, 0.8) for _ in range(10)]

def run_verdict(blocks, trust_map):
    return [{"glyph": block["glyph"], "trust": trust} for block, trust in zip(blocks, trust_map)]

def calculate_entropy(glyphs):
    glyph_freq = {g: glyphs.count(g) for g in set(glyphs)}
    total = len(glyphs)
    entropy = -sum((freq / total) * math.log2(freq / total) for freq in glyph_freq.values())
    return entropy / math.log2(len(set(glyphs))) if len(set(glyphs)) > 1 else 0

def narrate(glyphs):
    archetypes = {"⟁": "Initiator", "⧉": "Architect", "⨀": "Seer", "◌": "Void"}
    entropy = calculate_entropy(glyphs)
    mood = "foreboding" if entropy < 0.5 else "revelatory"
    narrative = f"A {mood} sequence unfolds through " + ", ".join(archetypes.get(g, "Unknown") for g in glyphs)
    return narrative

def arc_scan(verdict_log):
    transitions = []
    for i in range(1, len(verdict_log)):
        prev, curr = verdict_log[i-1], verdict_log[i]
        if curr["trust"] > prev["trust"]:
            transitions.append("Ascension")
        elif curr["trust"] < prev["trust"]:
            transitions.append("Dissonance")
        else:
            transitions.append("Echo")
    return transitions

def emotional_shadow(glyphs, trust_curve):
    mood_map = {"⟁": "courage", "⧉": "curiosity", "⨀": "clarity", "◌": "uncertainty"}
    average_trust = sum(trust_curve) / len(trust_curve)
    mood_weights = [mood_map.get(g, "neutral") for g in glyphs]
    tone = "resolute" if average_trust > 0.7 else "doubtful"
    return f"The glyph chorus whispers with {tone} moods of " + ", ".join(set(mood_weights))

def calculate_fog_pulse(trust_curve):
    mean = sum(trust_curve) / len(trust_curve)
    variance = sum((t - mean)**2 for t in trust_curve) / len(trust_curve)
    return min(1.0, math.sqrt(variance))

# 🎨 Verdict Visualization
def update_plots(axs, canvas, verdicts):
    glyphs = [v["glyph"] for v in verdicts]
    trust_curve = [v["trust"] for v in verdicts]
    forecast = ["⧂", "⧫", "⦿"]
    arcs = arc_scan(verdicts)

    # Wave Panel
    axs[0].clear()
    axs[0].plot(range(len(trust_curve)), trust_curve, color="cyan")
    axs[0].set_title("⧉ Ghost Resonance Pattern")

    # Trail Panel
    glyphs_history = glyphs * 2
    axs[1].clear()
    axs[1].text(0.1, 0.5, " ".join(glyphs_history), fontsize=18, fontfamily="monospace")
    axs[1].set_title("Glyph Trail")
    axs[1].axis("off")

    # Constellation Panel
    swarm_nodes = [{"pos": (random.random(), random.random()), "glyph": g} for g in glyphs[:3]]
    glyph_colors = {"⟁": "red", "⧉": "green", "⨀": "blue", "◌": "yellow"}

    axs[2].clear()
    for node in swarm_nodes:
        x, y = node["pos"]
        g = node["glyph"]
        axs[2].text(x, y, g, fontsize=14, color=glyph_colors.get(g, "white"),
                    transform=axs[2].transAxes, ha='center', va='center')
    axs[2].set_title("Swarm Glyph Constellation")
    axs[2].axis("off")

    # Matrix Panel
    counts = {g: glyphs.count(g) for g in ["⟁", "⧉", "⨀", "◌"]}
    data = [[counts.get(g, 0)] for g in ["⟁", "⧉", "⨀", "◌"]]
    sns.heatmap(data, annot=True, cmap="magma", yticklabels=list(counts.keys()), cbar=False, ax=axs[3])
    axs[3].set_title("Sigil Memory Matrix")

    # Arc Shadow Suite Console
    narration = narrate(glyphs)
    shadow = emotional_shadow(glyphs, trust_curve)
    fog_density = calculate_fog_pulse(trust_curve)
    forecast_str = " ".join(forecast)

    axs[4].clear()
    axs[4].text(0.05, 0.8, f"🔄 Arc Transitions:\n• " + "\n• ".join(arcs), fontsize=10, color="turquoise")
    axs[4].text(0.05, 0.5, f"🌫 Emotional Shadow: {shadow}", fontsize=10, color="orchid")
    axs[4].text(0.05, 0.3, f"🜁 Fog Pulse Intensity: {fog_density:.2f}", fontsize=9, color="gray")
    axs[4].text(0.05, 0.1, f"🜄 Forecast Glyphs: {forecast_str}", fontsize=10, color="gray")
    axs[4].axis("off")
    axs[4].set_title("Arc Shadow Suite Console")

    canvas.draw()

# ⏳ Continuous GUI Updater
def periodic_update(canvas, axs):
    blocks = scan_memory()
    trust_map = initialize_trust()
    verdicts = run_verdict(blocks, trust_map)
    update_plots(axs, canvas, verdicts)
    canvas.get_tk_widget().after(2000, lambda: periodic_update(canvas, axs))  # Every 2s

# 🪟 GUI Launcher
def launch_gui():
    window = tk.Tk()
    window.title("Glyph Verdict Engine")
    window.geometry("1200x840")
    window.configure(bg="#0a0a0a")

    fig, axs = plt.subplots(5, 1, figsize=(10, 20), constrained_layout=True)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    periodic_update(canvas, axs)
    window.mainloop()

# 🚀 Main Execution
if __name__ == "__main__":
    speak("Glyph Verdict Engine initialized.")
    print("\n🔮 Starting Glyph Verdict Engine\n")
    launch_gui()
    speak("Verdict engine complete. Interface closed.")
    print("\n✅ Interface closed.\n")

