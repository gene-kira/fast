# 🔮 MagicBox - Full One-Click AI Companion
import sys, os, json, subprocess, importlib

# 📦 Auto-install packages if missing
def ensure(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

pyttsx3 = ensure("pyttsx3")
import tkinter as tk

# 💾 Memory file path
MEMORY_FILE = "magic_profile.json"

# 🧠 Load saved memory
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

# 💾 Save memory to disk
def save_memory(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

class MagicBoxEZ:
    def __init__(self):
        self.memory = load_memory()
        self.engine = pyttsx3.init()
        self.root = tk.Tk()
        self.root.title("MagicBox 📦")
        self.root.configure(bg="#1B1B1B")
        self.root.geometry("500x300")

        self.label = tk.Label(self.root, text="🧙 Welcome! Choose a spell or type your own:", fg="#F0E6D2", bg="#1B1B1B", font=("Arial", 14))
        self.label.pack(pady=10)

        self.entry = tk.Entry(self.root, width=40, font=("Arial", 13))
        self.entry.pack(pady=8)
        if "last_spell" in self.memory:
            self.entry.insert(0, self.memory["last_spell"])

        btn_frame = tk.Frame(self.root, bg="#1B1B1B")
        btn_frame.pack(pady=5)

        # 🕹️ Preset buttons
        spells = ["Heal", "Fireball", "Shadowbind"]
        for s in spells:
            tk.Button(btn_frame, text=s, width=12, command=lambda spell=s: self.cast_spell(spell),
                      bg="#2E2E2E", fg="#F0E6D2", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

        tk.Button(self.root, text="🪄 Cast Typed Spell", command=self.cast_typed_spell,
                  bg="#3A3A3A", fg="#F0E6D2", font=("Arial", 12)).pack(pady=15)

        self.status = tk.Label(self.root, text="", fg="#B0FFC0", bg="#1B1B1B", font=("Arial", 11))
        self.status.pack()

        self.root.mainloop()

    def cast_typed_spell(self):
        spell = self.entry.get().strip()
        if spell:
            self.cast_spell(spell)

    def cast_spell(self, spell):
        spell_lc = spell.lower()
        if "heal" in spell_lc:
            response = "✨ Healing activated. You feel better already."
        elif "fireball" in spell_lc:
            response = "🔥 Fireball launched!"
        elif "shadow" in spell_lc:
            response = "🕶️ Shadowbind engaged. You blend with the shadows."
        else:
            response = f"You cast: {spell}"

        # 🔊 Speak response
        self.engine.say(response)
        self.engine.runAndWait()

        # 💾 Save to memory
        self.memory["last_spell"] = spell
        save_memory(self.memory)

        self.label.config(text=response)
        self.status.config(text=f"🧠 Spell remembered: '{spell}'")

MagicBoxEZ()

