# magicbox_asi.py
# ASI Agent + MagicBox GUI | Friendly One-Click Edition 🧙‍♂️

import importlib.util
import subprocess
import sys

# 🛠️ Autoloader for required packages (psutil, pillow)
def autoload(packages):
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
autoload(["psutil", "Pillow"])

# 🌐 Imports after autoload
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import psutil
import threading
import platform
import time

# ⚙️ Core Agent class
class AgentManager:
    def __init__(self):
        self.gaming_mode = False
        self.streamer_mode = False

    def activate_gaming_mode(self):
        self.gaming_mode = True
        print("🕹️ Gaming Mode Activated")
        # Mock behavior: print CPU load
        print(f"CPU Usage: {psutil.cpu_percent()}%")

    def deactivate_gaming_mode(self):
        self.gaming_mode = False
        print("🕹️ Gaming Mode Deactivated")

    def activate_streamer_mode(self):
        self.streamer_mode = True
        print("📺 Streamer Mode Activated")
        # Mock behavior: hide notifications (visual only here)

    def deactivate_streamer_mode(self):
        self.streamer_mode = False
        print("📺 Streamer Mode Deactivated")

# 🖥️ MagicBox GUI
class MagicBoxApp:
    def __init__(self, root):
        self.root = root
        self.agent = AgentManager()
        self.root.title("MagicBox ASI Agent 🌟")
        self.root.geometry("400x300")
        self.root.configure(bg="#20242c")

        # 🧙 Welcome Label
        self.title_label = tk.Label(root, text="MagicBox Edition", font=("Segoe UI", 18), fg="cyan", bg="#20242c")
        self.title_label.pack(pady=10)

        # 🕹️ Gaming Mode Toggle
        self.gaming_btn = tk.Button(root, text="Activate Gaming Mode", font=("Segoe UI", 12),
                                    bg="#2e8b57", fg="white", command=self.toggle_gaming_mode)
        self.gaming_btn.pack(pady=10)

        # 📺 Streamer Mode Toggle
        self.streamer_btn = tk.Button(root, text="Activate Streamer Mode", font=("Segoe UI", 12),
                                      bg="#8b0000", fg="white", command=self.toggle_streamer_mode)
        self.streamer_btn.pack(pady=10)

        # 📊 System Info
        self.status_label = tk.Label(root, text="Status: Idle", font=("Segoe UI", 10), fg="lightgray", bg="#20242c")
        self.status_label.pack(pady=20)

        self.update_stats()

    def toggle_gaming_mode(self):
        if not self.agent.gaming_mode:
            self.agent.activate_gaming_mode()
            self.gaming_btn.config(text="Deactivate Gaming Mode", bg="#556b2f")
            self.status_label.config(text="Status: Gaming Mode On")
        else:
            self.agent.deactivate_gaming_mode()
            self.gaming_btn.config(text="Activate Gaming Mode", bg="#2e8b57")
            self.status_label.config(text="Status: Idle")

    def toggle_streamer_mode(self):
        if not self.agent.streamer_mode:
            self.agent.activate_streamer_mode()
            self.streamer_btn.config(text="Deactivate Streamer Mode", bg="#4b0000")
            self.status_label.config(text="Status: Streamer Mode On")
        else:
            self.agent.deactivate_streamer_mode()
            self.streamer_btn.config(text="Activate Streamer Mode", bg="#8b0000")
            self.status_label.config(text="Status: Idle")

    def update_stats(self):
        def update_loop():
            while True:
                usage = psutil.cpu_percent()
                self.status_label.config(text=f"CPU Usage: {usage}%")
                time.sleep(2)
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

# 🚀 Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxApp(root)
    root.mainloop()

