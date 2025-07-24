# chart_visuals.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class GlyphChartFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#2c2c2c")
        self.signal_data = []

    def setup_chart(self):
        for widget in self.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title("Glyph Signal Confidence", color="white")
        ax.set_xlabel("Probe #", color="white")
        ax.set_ylabel("Confidence", color="white")
        ax.set_facecolor("#2c2c2c")
        fig.patch.set_facecolor("#2c2c2c")
        ax.tick_params(colors="white")

        if self.signal_data:
            ax.plot(self.signal_data, color="#9ae6b4", marker="o")

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack()

