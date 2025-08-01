# === 🧰 Autoloader Setup ===
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import random
    from collections import deque, defaultdict
    import ast
except ImportError as e:
    print("Missing:", e)
    raise SystemExit("Please install required modules.")

# === 🧠 Thought Engine using AST ===
class ThoughtProcessor:
    def __init__(self):
        self.log = []

    def analyze(self, data):
        try:
            tree = ast.parse(data)
            nodes = [type(n).__name__ for n in ast.walk(tree)]
            self.log.append(nodes)
            return f"Thoughts: {', '.join(nodes)}"
        except:
            return "Thoughts: Non-code or abstract pattern detected"

# === 🔐 Threat Engine ===
class ThreatDetector:
    def __init__(self):
        self.history = deque(maxlen=10)
        self.alert = False

    def scan(self, block_id):
        self.history.append(block_id)
        if self.history.count(block_id) > 3 or self._spike_detected():
            self.alert = True
            return "⚠️ Threat Detected: Spike or repeat access"
        else:
            self.alert = False
            return "✅ Normal Access"

    def _spike_detected(self):
        if len(self.history) < 3:
            return False
        diffs = [abs(self.history[i] - self.history[i-1]) for i in range(1, len(self.history))]
        return any(d > 50 for d in diffs)

# === 🗃️ RAID Core Logic ===
class PredictiveRaid:
    def __init__(self, disks=4, cache_size=10):
        self.disks = [defaultdict(str) for _ in range(disks)]
        self.cache = deque(maxlen=cache_size)
        self.read_history = deque(maxlen=50)
        self.thinker = ThoughtProcessor()
        self.threat = ThreatDetector()
        self.mood = "Calm"

    def set_mood(self, mood):
        self.mood = mood

    def write(self, block_id, data):
        disk_id = block_id % len(self.disks)
        parity = self._make_parity(data)
        self.disks[disk_id][block_id] = data
        self.cache.append((block_id, data))
        thoughts = self.thinker.analyze(data)
        return f"📝 Write → Block {block_id} | Disk {disk_id} | Parity: {parity}\n{thoughts}"

    def read(self, block_id):
        self.read_history.append(block_id)
        threat_msg = self.threat.scan(block_id)
        prediction = self._predict_next()
        disk_id = block_id % len(self.disks)
        data = self.disks[disk_id].get(block_id, "<missing>")
        return f"🔎 Read ← Block {block_id} | Disk {disk_id}\nPrediction: {prediction}\nData: {data}\n{threat_msg}"

    def _make_parity(self, data):
        return ''.join(chr(ord(c)^1) for c in data)

    def _predict_next(self):
        mood_bias = {"Calm": 1, "Curious": 2, "Agitated": 5, "Paranoid": 8}
        if len(self.read_history) < 2:
            return random.randint(0, 100)
        recent = list(self.read_history)[-2:]
        delta = recent[-1] - recent[-2]
        offset = mood_bias.get(self.mood, 1)
        return recent[-1] + delta * offset

# === 🎨 MagicBox GUI ===
class MagicBoxApp:
    def __init__(self, master):
        self.master = master
        master.title("🌌 EchoStream: Ascension Core Edition")
        master.configure(bg="#1b1b1b")
        self.raid = PredictiveRaid()

        # Fonts
        self.font = ("Consolas", 14)
        self.title_font = ("Consolas", 18, "bold")

        tk.Label(master, text="🧠 MagicBox Console", font=self.title_font, fg="#00ffff", bg="#1b1b1b").pack(pady=10)

        # Inputs
        self.block_entry = tk.Entry(master, font=self.font, width=10)
        self.block_entry.pack()
        self.block_entry.insert(0, "1")

        self.data_entry = tk.Entry(master, font=self.font, width=30)
        self.data_entry.pack()
        self.data_entry.insert(0, "alpha = 42")

        # Mood Selection
        tk.Label(master, text="🌀 Select Mood:", font=self.font, fg="white", bg="#1b1b1b").pack(pady=5)
        self.mood_selector = ttk.Combobox(master, font=self.font, values=["Calm", "Curious", "Agitated", "Paranoid"])
        self.mood_selector.pack()
        self.mood_selector.current(0)

        # Buttons
        tk.Button(master, text="📝 Write Block", font=self.font, bg="#333", fg="white",
                  command=self.write_block).pack(pady=5)

        tk.Button(master, text="🔎 Read Block", font=self.font, bg="#333", fg="white",
                  command=self.read_block).pack(pady=5)

        tk.Button(master, text="🚨 Override Trigger", font=self.font, bg="#880000", fg="white",
                  command=self.override_trigger).pack(pady=5)

        # Threat Panel
        self.threat_panel = tk.Label(master, text="🛡️ Status: No Threat", font=self.font,
                                     fg="#00ff00", bg="#1b1b1b")
        self.threat_panel.pack(pady=5)

        # Output Console
        self.output_label = tk.Label(master, text="", font=self.font, fg="white", bg="#1b1b1b",
                                     justify=tk.LEFT, wraplength=600)
        self.output_label.pack(pady=10)

    def update_mood(self):
        mood = self.mood_selector.get()
        self.raid.set_mood(mood)

    def write_block(self):
        self.update_mood()
        try:
            block_id = int(self.block_entry.get())
            data = self.data_entry.get()
            result = self.raid.write(block_id, data)
            self.output_label.config(text=result)
        except ValueError:
            messagebox.showerror("Input Error", "Block ID must be integer.")

    def read_block(self):
        self.update_mood()
        try:
            block_id = int(self.block_entry.get())
            result = self.raid.read(block_id)
            self.output_label.config(text=result)
            if "⚠️" in result:
                self.threat_panel.config(text="🛡️ Status: Threat Detected", fg="red")
            else:
                self.threat_panel.config(text="🛡️ Status: No Threat", fg="#00ff00")
        except ValueError:
            messagebox.showerror("Input Error", "Block ID must be integer.")

    def override_trigger(self):
        self.raid.set_mood("Paranoid")
        self.threat_panel.config(text="🔓 Override Active → Ascension Mode", fg="yellow")
        self.output_label.config(text="🔊 Override triggered. Fallback logic initiated.\nMood set to Paranoid.")

# === 🚀 Launcher ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxApp(root)
    root.mainloop()

