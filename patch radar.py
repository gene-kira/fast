import tkinter as tk
from tkinter import ttk
import math
import random

# ✔️ Autoload Check
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 📌 Persona Glyphs
PERSONAS = ['logician', 'mystic', 'reformer']
PATCH_SCORES = {'logician': 0.7, 'mystic': 0.6, 'reformer': 0.8}

# 📶 Influence Matrix
INFLUENCE_MATRIX = {
    'logician': {'mystic': 0.3, 'reformer': 0.4},
    'mystic': {'logician': 0.5, 'reformer': 0.2},
    'reformer': {'logician': 0.6, 'mystic': 0.3}
}

# 🎯 Adjust vote based on influence
def adjust_vote(base_score, persona, patch_scores):
    total_influence = 0
    adjusted = 0
    for other, score in patch_scores.items():
        influence = INFLUENCE_MATRIX.get(persona, {}).get(other, 0)
        adjusted += influence * score
        total_influence += influence
    return (base_score + adjusted) / (1 + total_influence)

# 🌐 Radar Plot
def draw_radar(scores):
    angles = [n / float(len(scores)) * 2 * math.pi for n in range(len(scores))]
    values = list(scores.values())
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scores.keys())
    return fig

# 🖥️ Main GUI
class GlyphGUI:
    def __init__(self, root):
        self.root = root
        root.title("GlyphGuard Patch Evaluator")

        ttk.Label(root, text="Click to Forecast Patch").pack(pady=10)
        ttk.Button(root, text="Run Patch Radar", command=self.run_patch).pack()

        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack()

    def run_patch(self):
        adjusted_scores = {p: adjust_vote(PATCH_SCORES[p], p, PATCH_SCORES) for p in PERSONAS}
        fig = draw_radar(adjusted_scores)

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        radar_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        radar_canvas.draw()
        radar_canvas.get_tk_widget().pack()

# 🚀 One-Click Launch
def main():
    root = tk.Tk()
    app = GlyphGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

