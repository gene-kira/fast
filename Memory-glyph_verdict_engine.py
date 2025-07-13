# glyph_verdict_engine.py
# One-file autoloading Glyph Verdict Engine with GUI and voice

import tkinter as tk
from tkinter import messagebox
import pyttsx3

# ------------------- Auto-Init Voice Engine -------------------

def init_voice():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        print("🔇 Voice engine error:", e)
        return None

voice = init_voice()

def speak(text):
    if voice:
        voice.say(text)
        voice.runAndWait()

# ------------------- Core Logic -------------------

def scan_memory():
    speak("Scanning memory blocks.")
    print("🔍 Scanning memory blocks...")
    blocks = [
        {'id': 1, 'size': 64, 'status': 'clean'},
        {'id': 2, 'size': 128, 'status': 'unknown'},
        {'id': 3, 'size': 32, 'status': 'rogue'}
    ]
    for b in blocks:
        print(f"🧱 Block {b['id']} | {b['size']}MB | Status: {b['status']}")
    return blocks

def initialize_trust():
    speak("Initializing trust curves.")
    print("📈 Trust curves aligned.")
    return {'clean': 0.9, 'unknown': 0.5, 'rogue': 0.1}

def run_verdict(blocks, trust_map):
    speak("Running verdict evaluation.")
    print("⚖️ Evaluating verdicts...")
    verdicts = []
    for b in blocks:
        trust = trust_map[b['status']]
        if trust < 0.3:
            action = 'Kill'
            speak(f"Verdict: Kill Block {b['id']}.")
        elif trust > 0.7:
            action = 'Keep'
            speak(f"Verdict: Keep Block {b['id']}.")
        else:
            action = 'Observe'
            speak(f"Verdict: Observe Block {b['id']}.")
        verdicts.append((b['id'], action))
    return verdicts

# ------------------- GUI -------------------

def launch_gui():
    blocks = scan_memory()
    trust_map = initialize_trust()
    verdicts = run_verdict(blocks, trust_map)

    window = tk.Tk()
    window.title("Glyph Verdict Engine")
    window.geometry("600x420")
    window.configure(bg="#0a0a0a")

    tk.Label(window, text="🔮 Glyph Verdict Engine",
             font=("Courier", 18), fg="#00ffaa", bg="#0a0a0a").pack(pady=10)
    frame = tk.Frame(window, bg="#0a0a0a")
    frame.pack(pady=10)

    for b, v in zip(blocks, verdicts):
        color = {"clean": "#00ff00", "unknown": "#ffff00", "rogue": "#ff0000"}[b['status']]
        label = tk.Label(frame,
                         text=f"🧱 Block {b['id']} | {b['size']}MB | {b['status'].capitalize()}",
                         font=("Courier", 12), fg=color, bg="#0a0a0a")
        label.pack(anchor="w")

        btn = tk.Button(frame, text=f"Verdict: {v[1]}", bg="#333333", fg="#ffffff",
                        command=lambda b=b, v=v: handle_verdict(b, v))
        btn.pack(anchor="w", pady=4)

    window.mainloop()

def handle_verdict(block, verdict):
    if verdict[1] == 'Kill':
        speak(f"Executing kill on Block {block['id']}.")
        messagebox.showinfo("Verdict", f"🗡️ Block {block['id']} terminated.")
    elif verdict[1] == 'Keep':
        speak(f"Block {block['id']} retained.")
        messagebox.showinfo("Verdict", f"🛡️ Block {block['id']} retained.")
    else:
        speak(f"Block {block['id']} under observation.")
        messagebox.showinfo("Verdict", f"🌀 Block {block['id']} observed.")

# ------------------- Start -------------------

if __name__ == "__main__":
    speak("Glyph Verdict Engine initialized.")
    print("\n🔮 Starting Glyph Verdict Engine\n")
    launch_gui()
    speak("Verdict engine complete. Interface closed.")
    print("\n✅ Interface closed.\n")

