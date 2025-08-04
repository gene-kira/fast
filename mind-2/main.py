# main.py — Overmind Launcher
import os
import threading
from gui.overmind_gui import launch_gui
from core.threat_loop import run_threat_assessment
from core.emotion_engine import initialize_emotions
from system.memory import boot_memory_core

def initialize_system():
    print("[Overmind] Initializing memory core...")
    boot_memory_core()

    print("[Overmind] Activating emotional simulation...")
    initialize_emotions()

    print("[Overmind] Spinning up threat assessment loop...")
    threading.Thread(target=run_threat_assessment, daemon=True).start()

    print("[Overmind] Launching GUI interface...")
    launch_gui()

if __name__ == "__main__":
    os.system("title Overmind AI System")
    print("╔═════════════════════════════╗")
    print("║  OVERMIND CORE ONLINE       ║")
    print("╚═════════════════════════════╝")
    initialize_system()

