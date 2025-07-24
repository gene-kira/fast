# node_mesh.py

import tkinter as tk
import random

class MeshRadarFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, width=500, height=400, bg="#2c2c2c")
        self.canvas = tk.Canvas(self, width=480, height=380, bg="#1b1b1b", highlightthickness=0)
        self.canvas.pack()
        self.nodes = []
        self.draw_nodes()

    def draw_nodes(self):
        for i in range(6):
            x, y = 70*i+30, random.randint(40, 300)
            node = self.canvas.create_oval(x, y, x+20, y+20, fill="#5c6bc0")
            self.nodes.append(node)

    def pulse_nodes(self):
        for node in self.nodes:
            self.canvas.itemconfig(node, fill=random.choice(["#9ae6b4", "#ffdd57", "#ff6b6b"]))

