# 🔮 MagicBox EZ Edition – One-Click, Autoload, Voice, and GUI

import sys, subprocess, importlib, os, json

# 📦 Auto-install required packages
def ensure(package_name):
    try:
        return importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(package_name)

pyttsx3 = ensure("pyttsx3")
tk = ensure("tkinter")  # typically built-in

# 💾 Memory system
MEMORY_FILE = "magic_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

# 🧙 MagicBox App
class MagicBoxEZ:
    def __init__(self):
        self.memory = load_memory()
        self.engine = pyttsx3.init()
        self.root = tk.Tk()
        self.root.title("MagicBox 📦")
        self.root.configure(bg="#1B1B1B")
        self.root.geometry("500x260")

        # 🪄 Welcome
        self.label = tk.Label(self.root, text="Type or tap a spell below:",
                              fg="#F0E6D2", bg="#1B1B1B", font=("Arial", 14))
        self.label.pack(pady=12)

        # ✏️ Text input
        self.entry = tk.Entry(self.root, width=40, font=("Arial", 13))
        self.entry.pack()
        self.entry.insert(0, self.memory.get("last_spell", ""))

        # 🔘 Preset spell buttons
        btn_frame = tk.Frame(self.root, bg="#1B1B1B")
        btn_frame.pack(pady=10)
        spells = ["Heal", "Fireball", "Shadowbind"]
        for spell in spells:
            tk.Button(btn_frame, text=spell, width=12,
                      command=lambda s=spell: self.cast_spell(s),
                      bg="#2E2E2E", fg="#F0E6D2", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

        # 🔮 Cast typed spell
        tk.Button(self.root, text="🪄 Cast Typed Spell", command=self.cast_typed,
                  bg="#3A3A3A", fg="#F0E6D2", font=("Arial", 12)).pack(pady=8)

        # 📜 Status
        self.status = tk.Label(self.root, text="", fg="#B0FFC0", bg="#1B1B1B", font=("Arial", 11))
        self.status.pack()

        self.root.mainloop()

    def cast_typed(self):
        spell = self.entry.get().strip()
        if spell:
            self.cast_spell(spell)

    def cast_spell(self, spell):
        spell_lower = spell.lower()
        if "heal" in spell_lower:
            response = "✨ Healing activated. You feel better."
        elif "fireball" in spell_lower:
            response = "🔥 Fireball launched!"
        elif "shadow" in spell_lower:
            response = "🕶️ Shadowbind engaged. Stealth mode on."
        else:
            response = f"You cast: {spell}"

        self.engine.say(response)
        self.engine.runAndWait()

        self.label.config(text=response)
        self.status.config(text=f"🧠 Remembered spell: {spell}")
        self.memory["last_spell"] = spell
        save_memory(self.memory)

MagicBoxEZ()

