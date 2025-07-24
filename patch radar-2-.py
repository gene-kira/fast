import tkinter as tk
from tkinter import ttk
import math
import random

# 📦 Autoloader for required libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 🌐 Glyphs + Influence Matrix
PERSONAS = ['logician', 'mystic', 'reformer']
PATCH_HISTORY = [
    {
        'id': 'Patch_001',
        'scores': {'logician': 0.7, 'mystic': 0.6, 'reformer': 0.8},
        'mood': 'innovation',
        'alignment': {'logician': ['reformer'], 'mystic': []}
    },
    {
        'id': 'Patch_002',
        'scores': {'logician': 0.65, 'mystic': 0.7, 'reformer': 0.75},
        'mood': 'risk',
        'alignment': {'reformer': ['mystic'], 'logician': []}
    }
]

MOOD_COLORS = {
    'innovation': '#7fdfff',
    'risk': '#ff8787',
    'stability': '#aaffaa',
    'chaos': '#ffaaff'
}

INFLUENCE_MATRIX = {
    'logician': {'mystic': 0.3, 'reformer': 0.4},
    'mystic': {'logician': 0.5, 'reformer': 0.2},
    'reformer': {'logician': 0.6, 'mystic': 0.3}
}

# 🧮 Voting logic
def adjust_vote(base_score, persona, patch_scores):
    total_influence = 0
    adjusted = 0
    for other, score in patch_scores.items():
        influence = INFLUENCE_MATRIX.get(persona, {}).get(other, 0)
        adjusted += influence * score
        total_influence += influence
    return (base_score + adjusted) / (1 + total_influence)

# 🌌 Radar visualization
def draw_radar(scores):
    angles = [n / float(len(scores)) * 2 * math.pi for n in range(len(scores))]
    values = list(scores.values())
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='navy')
    ax.fill(angles, values, 'deepskyblue', alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scores.keys())
    return fig

# 🧿 Main GUI with playback
class GlyphGUI:
    def __init__(self, root):
        self.root = root
        root.title("GlyphGuard Playback Interface")

        ttk.Label(root, text="Patch Forecast Playback", font=("Segoe UI", 12)).pack(pady=10)
        ttk.Button(root, text="Run Current Patch", command=self.run_patch).pack(pady=5)
        ttk.Button(root, text="Play Glyph Memory", command=lambda: self.playback_step(0)).pack(pady=5)

        self.canvas_frame = tk.Frame(root, width=400, height=400)
        self.canvas_frame.pack()

        self.radar_overlay = tk.Canvas(self.canvas_frame, width=400, height=400, highlightthickness=0)
        self.radar_overlay.place(x=0, y=0)
        self.last_color = '#ffffff'

    def run_patch(self):
        scores = PATCH_HISTORY[0]['scores']
        adjusted = {p: adjust_vote(scores[p], p, scores) for p in PERSONAS}
        fig = draw_radar(adjusted)

        for widget in self.canvas_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        radar = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        radar.draw()
        radar.get_tk_widget().pack()

        self.glyph_positions = self.get_glyph_positions(200, 200)
        self.trust_levels = {p: len(PATCH_HISTORY[0]['alignment'].get(p, [])) / len(PERSONAS) for p in PERSONAS}
        self.animate_ripples(step=0)

    def playback_step(self, index):
        if index >= len(PATCH_HISTORY):
            return

        patch = PATCH_HISTORY[index]
        scores = patch['scores']
        mood = patch['mood']
        new_color = MOOD_COLORS.get(mood, '#ffffff')

        self.fade_to_mood(self.last_color, new_color, step=0)
        self.last_color = new_color

        adjusted = {p: adjust_vote(scores[p], p, scores) for p in PERSONAS}
        fig = draw_radar(adjusted)

        for widget in self.canvas_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()

        radar = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        radar.draw()
        radar.get_tk_widget().pack()

        self.glyph_positions = self.get_glyph_positions(200, 200)
        self.trust_levels = {p: len(patch['alignment'].get(p, [])) / len(PERSONAS) for p in PERSONAS}
        self.animate_ripples(step=0)

        self.root.after(2000, lambda: self.playback_step(index + 1))

    def fade_to_mood(self, start_color, end_color, step=0, steps=20):
        def interp(c1, c2, s): return int(c1 + (c2 - c1) * s / steps)
        r1, g1, b1 = [int(start_color[i:i+2], 16) for i in (1,3,5)]
        r2, g2, b2 = [int(end_color[i:i+2], 16) for i in (1,3,5)]

        r, g, b = interp(r1, r2, step), interp(g1, g2, step), interp(b1, b2, step)
        new_color = f"#{r:02x}{g:02x}{b:02x}"
        self.canvas_frame.configure(bg=new_color)
        if step < steps:
            self.root.after(50, lambda: self.fade_to_mood(start_color, end_color, step + 1, steps))

    def get_glyph_positions(self, center_x, center_y):
        pos = {}
        radius = 100
        angle_step = 2 * math.pi / len(PERSONAS)
        for i, name in enumerate(PERSONAS):
            angle = i * angle_step
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            pos[name] = (x, y)
        return pos

    def animate_ripples(self, step=0):
        self.radar_overlay.delete("ripple")
        for persona, (x, y) in self.glyph_positions.items():
            trust = self.trust_levels.get(persona, 0.5)
            radius = 30 + step * 7
            color_intensity = int(150 + trust * 105)
            color = f"#00{color_intensity:02x}{color_intensity:02x}"
            self.radar_overlay.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline=color, width=2, tags="ripple"
            )
        if step < 8:
            self.root.after(120, lambda: self.animate_ripples(step + 1))

# 🚀 One-Click Start
def main():
    root = tk.Tk()
    app = GlyphGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

