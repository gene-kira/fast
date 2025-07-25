import tkinter as tk
from tkinter import messagebox, ttk
import psutil, pyttsx3, random, time, subprocess, sys
from datetime import datetime

# 🪄 Auto-install missing libraries
def autoload(libs):
    for lib in libs:
        try:
            __import__(lib)
        except ImportError:
            print(f"[MagicBox] Installing: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

required_libraries = ["psutil", "pyttsx3"]
autoload(required_libraries)

# 🧠 Spectral Optimizer AI
class SpectralOptimizer:
    def __init__(self):
        self.history = []
        self.optimization_log = []
        self.last_action_time = 0

    def record_metrics(self):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        timestamp = time.time()
        self.history.append({
            "time": timestamp,
            "memory": mem.percent,
            "available": mem.available,
            "cpu": cpu
        })

    def evaluate(self):
        if not self.history:
            return None
        recent = self.history[-1]
        mem = recent["memory"]
        cpu = recent["cpu"]
        now = time.time()

        if mem > 85 and now - self.last_action_time > 60:
            self.last_action_time = now
            self.optimization_log.append((now, "High memory detected. Suggest cleanup."))
            return "suggest_cleanup"

        elif cpu < 20 and mem < 30 and now - self.last_action_time > 120:
            self.last_action_time = now
            self.optimization_log.append((now, "Low usage. Reducing visual intensity."))
            return "reduce_visuals"

        return "idle"

# 👻 Visual Overlay Engine
class VisualOverlayEngine:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, bg="#2e004f", highlightthickness=0)
        self.canvas.place(relwidth=1, relheight=1)
        self.season = self.detect_season()
        self.overlays = []
        self.ghost = None
        self.active = True
        self.create_ghost()
        self.animate()

    def detect_season(self):
        month = datetime.now().month
        if month in [12, 1, 2]: return "winter"
        elif month in [6, 7, 8]: return "summer"
        elif month in [3, 4, 5]: return "spring"
        else: return "autumn"

    def create_ghost(self):
        self.ghost = self.canvas.create_oval(350, 100, 410, 140, fill="#cccccc", outline="#999999")

    def spawn_overlay(self):
        x = random.randint(0, 800)
        if self.season == "autumn":
            leaf = self.canvas.create_oval(x, -10, x+20, 10, fill="#c97f3a", outline="")
            self.overlays.append(leaf)
        elif self.season == "winter":
            flake = self.canvas.create_oval(x, -10, x+10, 0, fill="#bdf2ff", outline="")
            self.overlays.append(flake)

    def animate(self):
        self.spawn_overlay()
        for item in self.overlays:
            self.canvas.move(item, 0, 2)
        if self.ghost:
            self.canvas.move(self.ghost, random.choice([-1, 1]), 0)
        if self.active:
            self.canvas.after(50, self.animate)

# 🎙️ Butler Sass System
class ButlerSass:
    def __init__(self, root):
        self.engine = pyttsx3.init()
        self.voice_enabled = True
        self.sass_level = "medium"
        self.last_trigger = {"low": 0, "medium": 0, "high": 0}
        self.sass_quotes = {
            "low": ["System's calm. Almost suspiciously so."],
            "medium": ["Memory dancing at mid-load.", "Half-full RAM and half-hearted optimism."],
            "high": ["Your memory's on fire. Want marshmallows?", "Well well... RAM’s maxed again."]
        }
        self.speak("MagicBox is now managing your system. Sit back and let the sass flow.")
        self.speak("I’m watching your RAM like it owes me money.")

    def speak(self, quote):
        if self.voice_enabled:
            self.engine.say(quote)
            self.engine.runAndWait()

    def get_quote(self):
        usage = psutil.virtual_memory().percent
        now = time.time()
        if usage < 30 and now - self.last_trigger["low"] > 30:
            self.last_trigger["low"] = now
            return random.choice(self.sass_quotes["low"])
        elif usage < 70 and now - self.last_trigger["medium"] > 15:
            self.last_trigger["medium"] = now
            return random.choice(self.sass_quotes["medium"])
        elif usage >= 70 and now - self.last_trigger["high"] > 60:
            self.last_trigger["high"] = now
            return random.choice(self.sass_quotes["high"])
        return None

# 🧪 Debug Console
def build_debug_console(root, butler, optimizer):
    debug_frame = tk.Frame(root, bg="#1f1f1f")
    debug_frame.pack(side="bottom", fill="x")
    log_box = tk.Text(debug_frame, height=6, bg="#000000", fg="#39ff14", insertbackground="#39ff14")
    log_box.pack(fill="x")

    def update():
        mem = psutil.virtual_memory()
        quote = butler.get_quote()
        if quote:
            butler.speak(quote)
            log_box.insert("end", f"[{time.ctime()}] RAM: {mem.percent}% | {quote}\n")
            log_box.see("end")
        optimizer.record_metrics()
        action = optimizer.evaluate()
        if action != "idle":
            log_box.insert("end", f"[{time.ctime()}] {action}\n")
            log_box.see("end")
        root.after(15000, update)

    update()

# 🧓 Main Launcher with Buttons
def launch_magicbox():
    root = tk.Tk()
    root.title("MagicBox - 🧓 One Click Wonder")
    root.geometry("850x600")
    root.configure(bg="#2e004f")

    overlay = VisualOverlayEngine(root)
    butler = ButlerSass(root)
    optimizer = SpectralOptimizer()

    # 🧭 Add control buttons
    button_frame = tk.Frame(root, bg="#2e004f")
    button_frame.pack(pady=20)

    def toggle_voice():
        butler.voice_enabled = not butler.voice_enabled
        voice_btn.config(text="Voice: ON" if butler.voice_enabled else "Voice: OFF")
        if butler.voice_enabled:
            butler.speak("Voice re-enabled. Sass shall resume.")

    def set_sass(level):
        butler.sass_level = level
        butler.speak(f"Sass level set to {level}.")

    def start_optimization():
        butler.speak("Optimization started. Let's get this RAM in shape.")

    optimize_btn = tk.Button(button_frame, text="Start Optimization", font=("Arial", 14), bg="#4caf50", fg="white", command=start_optimization)
    optimize_btn.pack(side="left", padx=10)

    voice_btn = tk.Button(button_frame, text="Voice: ON", font=("Arial", 14), bg="#2196f3", fg="white", command=toggle_voice)
    voice_btn.pack(side="left", padx=10)

    sass_label = tk.Label(button_frame, text="Sass Level:", font=("Arial", 14), bg="#2e004f", fg="white")
    sass_label.pack(side="left", padx=(30, 5))

    sass_var = tk.StringVar(value="medium")
    sass_menu = ttk.OptionMenu(button_frame, sass_var, "medium", "low", "medium", "high", command=set_sass)
    sass_menu.pack(side="left")

    build_debug_console(root, butler, optimizer)

    messagebox.showinfo("MagicBox Active", "MagicBox is now managing your system.\nSit back and let the sass flow.")

    root.mainloop()

# 🔘 Start It All
if __name__ == "__main__":
    launch_magicbox()

