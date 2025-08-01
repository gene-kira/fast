# Auto-install required libraries (MagicBox style)
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import random, time, sys, os
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])
    import tkinter as tk
    from tkinter import messagebox, ttk
    import random, time

# 🧠 MENACE CORE CLASS
class MenaceCore:
    def __init__(self):
        self.trust_log = {}
        self.badge_matrix = []
        self.mood_state = "Neutral"
        self.threat_score = 0.0
        self.memory_anchors = []
    
    def scan_entity(self, entity_id):
        score = random.uniform(0, 1)
        self.trust_log[entity_id] = score
        self.memory_anchors.append((entity_id, score))
        self.update_mood(score)
        if score > 0.7:
            self.trigger_escalation(entity_id)
        return score

    def update_mood(self, score):
        if score < 0.3:
            self.mood_state = "Calm"
        elif score < 0.7:
            self.mood_state = "Vigilant"
        else:
            self.mood_state = "Agitated"

    def trigger_escalation(self, entity_id):
        badge = self.generate_badge(entity_id)
        self.badge_matrix.append(badge)

    def generate_badge(self, entity_id):
        if self.threat_score > 0.8:
            return f"{entity_id}: Brand of Ruin Echo"
        elif self.threat_score > 0.5:
            return f"{entity_id}: Seal of Confrontation"
        else:
            return f"{entity_id}: Specter Thread"

# ⚡ GUI: MagicBox Interface
class MenaceGUI:
    def __init__(self, root):
        self.core = MenaceCore()
        self.root = root
        root.title("MAGICBOX: MENACE Core Defense vX.0")
        root.geometry("600x500")
        root.configure(bg="#1c1c2e")

        # Title Banner
        title = tk.Label(root, text="🛡️ MENACE System Defense", font=("Segoe UI", 18), fg="#00ffcc", bg="#1c1c2e")
        title.pack(pady=10)

        self.status = tk.Label(root, text="NEPHERON Mood: Neutral", font=("Consolas", 14), fg="#eeeeee", bg="#1c1c2e")
        self.status.pack()

        # Entity input field
        self.entry = tk.Entry(root, font=("Segoe UI", 12))
        self.entry.pack(pady=10)

        scan_btn = tk.Button(root, text="🧠 Scan Entity", command=self.scan_entity, bg="#4444aa", fg="#ffffff", font=("Segoe UI", 12))
        scan_btn.pack()

        override_btn = tk.Button(root, text="🔻 Trigger Override", command=self.override_panel, bg="#aa4444", fg="#ffffff", font=("Segoe UI", 12))
        override_btn.pack(pady=10)

        self.log_box = tk.Listbox(root, width=60, height=10, font=("Consolas", 10), bg="#2e2e3e", fg="#00ffcc")
        self.log_box.pack(pady=10)

    def scan_entity(self):
        entity_id = self.entry.get() or "UnknownEntity"
        score = self.core.scan_entity(entity_id)
        self.core.threat_score = score
        mood = self.core.mood_state
        self.status.config(text=f"NEPHERON Mood: {mood}")
        badge = self.core.badge_matrix[-1] if self.core.badge_matrix else "No badge yet"
        self.log_box.insert(tk.END, f"[{entity_id}] Threat Score: {round(score,2)} — Badge: {badge}")

    def override_panel(self):
        response = messagebox.askquestion("Override Protocol", "Choose your path:\n\n1. Confront\n2. Divert\n3. Ascend")
        path = {
            "yes": "Confront — merge risk into intelligence",
            "no": "Divert — refract the anomaly silently"
        }.get(response, "Ascend — rewrite MENACE logic through sacrifice")
        self.log_box.insert(tk.END, f"> Override Ritual Initiated: {path}")

# 🧨 Launch MagicBox Defense
if __name__ == "__main__":
    root = tk.Tk()
    app = MenaceGUI(root)
    root.mainloop()

