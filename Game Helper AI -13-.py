# 🔮 MagicBox EZ - One-Click Friendly with Autoloader
import sys, subprocess, importlib, os, json

# 📦 Autoloader for required packages
def ensure(package_name):
    try:
        return importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(package_name)

# ✅ Load pyttsx3 (text-to-speech)
pyttsx3 = ensure("pyttsx3")
tk = ensure("tkinter")  # Usually built-in

# 💾 Save & load ritual memory
PROFILE_FILE = "magic_profile.json"

def load_memory():
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(data):
    try:
        with open(PROFILE_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

class MagicBoxSimple:
    def __init__(self):
        self.memory = load_memory()
        self.engine = pyttsx3.init()
        self.root = tk.Tk()
        self.root.title("MagicBox 📦")
        self.root.configure(bg="#1B1B1B")
        self.root.geometry("480x240")

        # 🧙‍♂️ Interface
        self.label = tk.Label(self.root, text="Choose or type your spell:", fg="#F0E6D2", bg="#1B1B1B", font=("Arial", 14))
        self.label.pack(pady=12)

        self.entry = tk.Entry(self.root, width=40, font=("Arial", 13))
        self.entry.pack()
        if "last_spell" in self.memory:
            self.entry.insert(0, self.memory["last_spell"])

        # 🎯 Preset Spell Buttons
        spell_frame = tk.Frame(self.root, bg="#1B1B1B")
        spell_frame.pack(pady=8)

        for spell in ["Heal", "Fireball", "Shadowbind"]:
            tk.Button(spell_frame, text=spell, width=12,
                      command=lambda s=spell: self.cast_spell(s),
                      bg="#2E2E2E", fg="#F0E6D2", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

        tk.Button(self.root, text="🪄 Cast Typed Spell", command=self.cast_typed,
                  bg="#3A3A3A", fg="#F0E6D2", font=("Arial", 12)).pack(pady=12)

        self.status = tk.Label(self.root, text="", fg="#C0FFC0", bg="#1B1B1B", font=("Arial", 11))
        self.status.pack()

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
            response = "🕶️ Shadowbind engaged. You vanish in silence."
        else:
            response = f"You cast: {spell}"

        self.engine.say(response)
        self.engine.runAndWait()
        self.label.config(text=response)
        self.status.config(text=f"🧠 Remembered spell: {spell}")
        self.memory["last_spell"] = spell
        save_memory(self.memory)

MagicBoxSimple()

