# === AUTOLOADER ===
try:
    import hashlib, json, random, datetime, os, tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    os.system("pip install tkinter")

# === MENACE CORE SYSTEM ===
class MENACEDefense:
    def __init__(self):
        self.trust_map = {}
        self.evolving_modules = []
        self.override_triggered = False
        self.logs = []

    def log_event(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")

    def assess_threat(self, entity_data):
        signature = hashlib.sha256(json.dumps(entity_data).encode()).hexdigest()
        trust_score = self.trust_map.get(signature, random.randint(0, 100))
        self.log_event(f"Assessing {entity_data['ip']} | Trust={trust_score}")
        return trust_score < 50

    def evolve_module(self):
        new_mod = {
            "id": random.randint(1000, 9999),
            "created": datetime.datetime.now().isoformat(),
            "code": f"# Self-generated module {random.randint(0,999999)}"
        }
        self.evolving_modules.append(new_mod)
        self.log_event("🧠 New defensive module synthesized.")

    def override_protocol(self, reason):
        self.override_triggered = True
        self.log_event(f"⚠️ Override triggered: {reason}")
        self.evolve_module()

# === GUI INTERFACE ===
class MENACEGUI:
    def __init__(self, brain: MENACEDefense):
        self.brain = brain
        self.root = tk.Tk()
        self.root.title("🛡️ MENACE Defense System")
        self.root.geometry("650x400")
        self.root.configure(bg='black')
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use("alt")
        style.configure("TButton", padding=6, relief="flat", background="#333")

        self.label = tk.Label(self.root, text="MENACE Status Console", fg="lime", bg="black", font=("Courier", 16))
        self.label.pack(pady=10)

        self.status_box = tk.Text(self.root, bg="black", fg="white", font=("Courier", 10), width=80, height=15)
        self.status_box.pack()

        self.button = ttk.Button(self.root, text="Run Defense Protocol", command=self.run_defense)
        self.button.pack(pady=10)

    def run_defense(self):
        intruder = {"ip": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}", "behavior": "unknown", "access_level": "admin"}
        threat = self.brain.assess_threat(intruder)
        if threat:
            self.brain.override_protocol("Threat behavior detected")
        self.update_logs()

    def update_logs(self):
        self.status_box.delete("1.0", tk.END)
        for log in self.brain.logs[-20:]:
            self.status_box.insert(tk.END, log + "\n")

    def launch(self):
        self.root.mainloop()

# === MAIN LAUNCHER ===
if __name__ == "__main__":
    brain = MENACEDefense()
    interface = MENACEGUI(brain)
    interface.launch()

