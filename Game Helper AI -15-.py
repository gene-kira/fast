# 🔮 MagicBox AI – Unified Build: Autoloader, Memory, Voice, Game Alerts, EZ GUI

import sys, subprocess, importlib, os, json, time, threading
from tkinter import Tk, Label, Entry, Button, Frame, messagebox

# 📦 Auto-install required packages
def ensure(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

pyttsx3 = ensure("pyttsx3")
try:
    import winsound
except ImportError:
    winsound = None  # Sound fallback for non-Windows systems

# 💾 File paths
MEMORY_FILE = "magic_profile.json"
LOG_FILE = "gamelog.txt"

# 🎙️ EchoDaemon – Class-Based Voice Modulation
class EchoDaemon:
    def __init__(self, engine, profile):
        self.engine = engine
        self.profile = profile
        self.configured = False

    def calibrate(self):
        cls = self.profile.get("class", "Wanderer")
        config = {
            "Warrior": {"rate": 160},
            "Mage": {"rate": 120},
            "Rogue": {"rate": 180},
            "Necromancer": {"rate": 100},
            "Summoner": {"rate": 140},
            "Wanderer": {"rate": 140}
        }
        rate = config.get(cls, config["Wanderer"])["rate"]
        self.engine.setProperty("rate", rate)
        self.configured = True

    def speak(self, text):
        if not self.configured:
            self.calibrate()
        self.engine.say(text)
        self.engine.runAndWait()

# 📘 Load/save player memory
def load_profile():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"name": "Adventurer", "class": "Wanderer", "last_spell": ""}

def save_profile(profile):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(profile, f)
    except:
        pass

# 🔮 MagicBox Main Class
class MagicBoxAI:
    def __init__(self):
        self.profile = load_profile()
        self.engine = pyttsx3.init()
        self.echo = EchoDaemon(self.engine, self.profile)

        self.root = Tk()
        self.root.title("MagicBox 🧙")
        self.root.configure(bg="#1B1B1B")
        self.root.geometry("540x330")

        self.label = Label(self.root, text=f"Welcome, {self.profile['name']} the {self.profile['class']}!",
                           fg="#F0E6D2", bg="#1B1B1B", font=("Arial", 14))
        self.label.pack(pady=10)

        self.entry = Entry(self.root, width=40, font=("Arial", 13))
        self.entry.pack()
        self.entry.insert(0, self.profile.get("last_spell", ""))

        # 🕹️ Preset buttons
        spells = {
            "Heal": "Restores health over time.",
            "Fireball": "Launches a fiery blast.",
            "Shadowbind": "Stealth and silence combo."
        }
        btn_frame = Frame(self.root, bg="#1B1B1B")
        btn_frame.pack(pady=10)

        for spell, tip in spells.items():
            btn = Button(btn_frame, text=spell, width=12,
                         command=lambda s=spell: self.cast_spell(s),
                         bg="#2E2E2E", fg="#F0E6D2", font=("Arial", 12))
            btn.pack(side="left", padx=5)
            btn.bind("<Enter>", lambda e, t=tip: self.label.config(text=f"💡 {t}"))
            btn.bind("<Leave>", lambda e: self.label.config(text=f"Welcome, {self.profile['name']} the {self.profile['class']}!"))

        Button(self.root, text="🪄 Cast Typed Spell", command=self.cast_typed,
               bg="#3A3A3A", fg="#F0E6D2", font=("Arial", 12)).pack(pady=10)

        self.status = Label(self.root, text="", fg="#B0FFC0", bg="#1B1B1B", font=("Arial", 11))
        self.status.pack()

        threading.Thread(target=self.monitor_log, daemon=True).start()
        self.root.mainloop()

    def cast_typed(self):
        spell = self.entry.get().strip()
        if spell:
            self.cast_spell(spell)

    def cast_spell(self, spell):
        spell_lc = spell.lower()
        if "heal" in spell_lc:
            response = "✨ Healing spell activated. You feel refreshed."
        elif "fireball" in spell_lc:
            response = "🔥 Fireball launched!"
        elif "shadow" in spell_lc:
            response = "🕶️ Shadowbind engaged. You vanish from sight."
        else:
            response = f"You cast: {spell}"

        self.echo.speak(response)
        self.label.config(text=response)
        self.status.config(text=f"🧠 Remembered spell: '{spell}'")

        self.profile["last_spell"] = spell
        save_profile(self.profile)

        # ✨ Visual feedback
        self.root.configure(bg="#222222")
        self.root.after(300, lambda: self.root.configure(bg="#1B1B1B"))

    def monitor_log(self):
        time.sleep(3)
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r") as f:
                    data = f.read().lower()
                    if "game started" in data or "loading world" in data:
                        if winsound:
                            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                        messagebox.showinfo("MagicBox", f"🟢 Game started! Welcome back, {self.profile['name']}.")
                        self.echo.speak(f"Game started. Welcome, {self.profile['name']} the {self.profile['class']}.")
                        self.status.config(text="🧙 Game world detected. Ritual memory is active.")
            except:
                pass

MagicBoxAI()

