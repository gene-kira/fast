# === 🧿 PART 1: Ritual Engines & Autoloader ===

# 🔧 Autoloader
import subprocess, sys
def autoload(pkg):
    try:
        globals()[pkg] = __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        globals()[pkg] = __import__(pkg)

# 📦 Required packages
for package in ["tkinter", "threading", "math", "time", "pyttsx3", "pygame"]:
    autoload(package)

# 🧠 Core Imports
import tkinter as tk
import threading, time, math, pyttsx3
import pygame

# 🔊 Voice Engine Initialization
pygame.mixer.init()
engine = pyttsx3.init()
voice_alert_active = True

# 🛰️ Agent Registry
agent_registry = {
    "Sentinel_007": {"role": "Sentinel", "trust": 0.91},
    "Archivist_314": {"role": "Archivist", "trust": 0.88},
    "Seeker_453": {"role": "Seeker", "trust": 0.67}
}

# ☄️ Comet Tail Trail for Glyph Decay
def create_comet_tail(canvas, x, y, color="#FF4444"):
    for i in range(6):
        trail = canvas.create_oval(x - i * 2, y + i * 2, x - i * 2 + 4, y + i * 2 + 4, fill=color, outline="")
        canvas.after(i * 40, lambda t=trail: canvas.delete(t))

# 🔄 Orbital Choreography + Labels
def animate_orbitals(canvas, agent_registry):
    angle_map = {}
    for i, aid in enumerate(agent_registry):
        angle_map[aid] = i * (360 / len(agent_registry))

    def orbit():
        while True:
            for aid in agent_registry:
                trust = agent_registry[aid]["trust"]
                if trust < 0.3:
                    continue

                role = agent_registry[aid]["role"]
                ring_offset = {"Sentinel": 100, "Archivist": 140, "Seeker": 180}.get(role, 150)
                angle_map[aid] = (angle_map[aid] + trust * 1.2) % 360
                angle = math.radians(angle_map[aid])
                x = 200 + ring_offset * math.cos(angle)
                y = 200 + ring_offset * math.sin(angle)

                canvas.delete(f"orbit_{aid}")
                canvas.delete(f"label_{aid}")
                canvas.create_text(x, y, text="🛰️", fill="cyan", font=("Helvetica", 14), tags=f"orbit_{aid}")
                canvas.create_text(x, y + 14, text=f"{aid}", fill="#DDDDDD", font=("Helvetica", 8), tags=f"label_{aid}")

                if trust < 0.5:
                    create_comet_tail(canvas, x, y)

            canvas.update()
            time.sleep(0.05)

    threading.Thread(target=orbit, daemon=True).start()

# 🌌 Glyph Constellation Viewer
def launch_constellation_view(agent_registry):
    viewer = tk.Toplevel()
    viewer.title("Glyph Constellation Viewer")
    canvas = tk.Canvas(viewer, width=600, height=600, bg="black")
    canvas.pack()

    center_x, center_y = 300, 300
    positions = {}
    radius_base = 200

    for i, aid in enumerate(agent_registry):
        angle = math.radians(i * (360 / len(agent_registry)))
        trust = agent_registry[aid]["trust"]
        r = radius_base - (trust * 60)
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        positions[aid] = (x, y)

        color = "#00FFAA" if trust > 0.8 else "#4444FF"
        glow = trust * 10
        canvas.create_oval(x - glow, y - glow, x + glow, y + glow, fill=color, outline="")
        canvas.create_text(x, y, text="🧿", fill="white", font=("Helvetica", 20))
        canvas.create_text(x, y + 20, text=f"{aid} ({agent_registry[aid]['role']})", fill="#CCCCCC", font=("Helvetica", 8))

    keys = list(agent_registry.keys())
    for i in range(len(keys) - 1):
        x1, y1 = positions[keys[i]]
        x2, y2 = positions[keys[i + 1]]
        canvas.create_line(x1, y1, x2, y2, fill="#888888", dash=(2, 4))

    canvas.create_text(center_x, 20, text="🪐 Trust Constellation Viewer", fill="white", font=("Helvetica", 14))

# === 🚀 PART 2: UI Launcher ===

def launch_seal_ui_with_voice():
    root = tk.Tk()
    root.title("Seal Ritual — Orbital Verdict System")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    # 🧿 Central Glyph
    glyph_core = canvas.create_oval(170, 170, 230, 230, fill="#2222FF", outline="white", width=3)
    canvas.create_text(200, 160, text="🧿_Glyph_003", fill="white", font=("Helvetica", 10))

    # 🔊 Voice Alert on Ritual Start
    global voice_alert_active
    if voice_alert_active:
        engine.say("Seal ritual initialized for 🧿 Glyph 003")
        engine.runAndWait()

    # 🔇 ACK Button
    def ack_alert():
        global voice_alert_active
        voice_alert_active = False
        print("🔇 Voice alert silenced")
    tk.Button(root, text="ACK Alert 🔇", command=ack_alert, bg="gray", fg="white", width=20).pack(pady=4)

    # 🌌 Toggle Constellation Viewer
    tk.Button(root, text="Constellation Map 🌌", bg="darkblue", fg="white",
              command=lambda: launch_constellation_view(agent_registry)).pack(pady=4)

    # 🎛️ Verdict Buttons per Agent
    VERDICTS = ["Home", "Tracking", "Self-Destruction"]
    for aid in agent_registry:
        frame = tk.Frame(root, bg="black")
        frame.pack(pady=1)
        tk.Label(frame, text=f"{aid}:", fg="white", bg="black").pack(side="left", padx=4)
        for verdict in VERDICTS:
            tk.Button(frame, text=verdict,
                      command=lambda a=aid, v=verdict: engine.say(f"{a} casts verdict {v}"),
                      bg="darkred" if verdict == "Self-Destruction" else "darkgreen" if verdict == "Home" else "darkblue",
                      fg="white", width=14).pack(side="left", padx=2)

    # 🛰️ Animate Orbitals
    animate_orbitals(canvas, agent_registry)

    root.mainloop()

# 🚀 Run the Ritual
if __name__ == "__main__":
    launch_seal_ui_with_voice()

