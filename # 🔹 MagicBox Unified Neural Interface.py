# 🔹 MagicBox: Unified Neural Interface Prototype
# 🧠 GUI + Voice Engine + Library Autoloader

import sys
import subprocess
import importlib
import tkinter as tk
import random
import math

# 🧩 Autoloader: Ensures required libraries are present
def autoload(packages):
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

autoload(["pyttsx3"])

# 🎤 Voice Engine Initialization
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
# Feel free to select a different voice index if desired
engine.setProperty('voice', voices[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

speak("Initializing MagicBox Interface")

# 🧠 Node Class Definition
class Node:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.radius = 3

    def move(self, width, height):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= width:
            self.dx *= -1
        if self.y <= 0 or self.y >= height:
            self.dy *= -1

    def draw(self):
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill="#00F7FF", outline=""
        )

# 🚀 GUI Launcher: MagicBox Edition
def launch_network_gui():
    root = tk.Tk()
    root.title("🧠 MagicBox Neural Interface")
    root.geometry("720x520")
    root.configure(bg="#0B0E1A")

    canvas_width = 700
    canvas_height = 460

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height,
                       bg="#0A0C1B", highlightthickness=0)
    canvas.pack(pady=30)

    # 🕸️ Node Swarm Initialization
    node_count = 40
    nodes = [Node(canvas, canvas_width, canvas_height) for _ in range(node_count)]

    def animate():
        canvas.delete("all")
        for node in nodes:
            node.move(canvas_width, canvas_height)
            node.draw()
        for i in range(node_count):
            for j in range(i + 1, node_count):
                n1, n2 = nodes[i], nodes[j]
                dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                if dist < 150:
                    canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00F7FF", width=1)
        root.after(30, animate)

    # 📣 Launch Greeting
    speak("Network lattice interface is now live")
    animate()
    root.mainloop()

# 🎯 Entry Point
if __name__ == "__main__":
    launch_network_gui()

