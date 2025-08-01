# === AUTOLOADER ===
try:
    import hashlib, json, random, datetime, os, tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    os.system("pip install tkinter")

# === MENACE DEFENSE SYSTEM CORE ===
class MENACEDefense:
    def __init__(self):
        self.trust_map = {}
        self.whitelist = set()
        self.blacklist = set()
        self.evolving_modules = []
        self.override_triggered = False
        self.logs = []
        self.last_signature = ""

    def log_event(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")

    def hash_entity(self, entity_data):
        return hashlib.sha256(json.dumps(entity_data).encode()).hexdigest()

    def whitelist_entity(self, signature):
        self.whitelist.add(signature)
        self.log_event(f"✅ Whitelisted: {signature}")

    def blacklist_entity(self, signature):
        self.blacklist.add(signature)
        self.log_event(f"⛔ Blacklisted: {signature}")

    def assess_threat(self, entity_data):
        signature = self.hash_entity(entity_data)
        self.last_signature = signature

        if signature in self.whitelist:
            self.log_event(f"🛡️ Trusted (Whitelisted): {entity_data['ip']}")
            return False
        if signature in self.blacklist:
            self.log_event(f"🚨 Blacklisted Threat: {entity_data['ip']}")
            return True

        trust_score = self.trust_map.get(signature, random.randint(0, 100))
        self.trust_map[signature] = trust_score
        self.log_event(f"🧪 Assessing {entity_data['ip']} | Trust={trust_score}")
        return trust_score < 50

    def evolve_module(self):
        new_mod = {
            "id": random.randint(1000, 9999),
            "created": datetime.datetime.now().isoformat(),
            "code": f"# Defensive module v{random.randint(1,1000)}"
        }
        self.evolving_modules.append(new_mod)
        self.log_event("🧠 New module synthesized.")

    def override_protocol(self, reason):
        self.override_triggered = True
        self.log_event(f"⚠️ Override triggered: {reason}")
        self.evolve_module()

# === MENACE GUI INTERFACE ===
class MENACEGUI:
    def __init__(self, brain: MENACEDefense):
        self.brain = brain
        self.root = tk.Tk()
        self.root.title("🛡️ MENACE Defense Sentinel")
        self.root.geometry("750x500")
        self.root.configure(bg='black')
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use("alt")
        style.configure("TButton", padding=6, relief="flat", background="#333")

        tk.Label(self.root, text="MENACE Defense Console", fg="lime", bg="black", font=("Courier", 18)).pack(pady=10)

        self.status_box = tk.Text(self.root, bg="black", fg="white", font=("Courier", 10), width=90, height=18)
        self.status_box.pack()

        tk.Frame(self.root, bg='black')
        btn_frame = tk.Frame(self.root, bg="black")
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Run Threat Assessment", command=self.run_defense).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Whitelist Last", command=self.whitelist_last).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Blacklist Last", command=self.blacklist_last).grid(row=0, column=2, padx=5)

    def run_defense(self):
        intruder = {
            "ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            "behavior": "unknown",
            "access_level": random.choice(["admin", "user", "guest"])
        }
        threat = self.brain.assess_threat(intruder)
        if threat:
            self.brain.override_protocol("Untrusted behavior detected")
        self.update_logs()

    def whitelist_last(self):
        self.brain.whitelist_entity(self.brain.last_signature)
        self.update_logs()

    def blacklist_last(self):
        self.brain.blacklist_entity(self.brain.last_signature)
        self.update_logs()

    def update_logs(self):
        self.status_box.delete("1.0", tk.END)
        for log in self.brain.logs[-20:]:
            self.status_box.insert(tk.END, log + "\n")

    def launch(self):
        self.root.mainloop()

# === MAIN MODULE ===
if __name__ == "__main__":
    brain = MENACEDefense()
    gui = MENACEGUI(brain)
    gui.launch()

