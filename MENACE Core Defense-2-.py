# Auto-install required libraries
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import random, time, sys, os, threading
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])
    import tkinter as tk
    from tkinter import messagebox, ttk
    import random, time
    import threading

# === MENACE: Core Intelligence Engine ===
class MenaceCore:
    def __init__(self):
        self.trust_log = {}
        self.badge_matrix = []
        self.mood_state = "Neutral"
        self.memory_anchors = []
        self.threat_score = 0.0

    def scan_entity(self, entity_id):
        score = random.uniform(0, 1)
        self.trust_log[entity_id] = score
        self.memory_anchors.append((entity_id, score))
        self.update_mood(score)
        if score > 0.75:
            self.trigger_escalation(entity_id)
        return score

    def update_mood(self, score):
        if score < 0.3:
            self.mood_state = "Clarity"
        elif score < 0.6:
            self.mood_state = "Vigilant"
        elif score < 0.85:
            self.mood_state = "Fury"
        else:
            self.mood_state = "Dread"

    def trigger_escalation(self, entity_id):
        badge = self.generate_badge(entity_id)
        self.badge_matrix.append(badge)

    def generate_badge(self, entity_id):
        mood = self.mood_state
        mood_map = {
            "Clarity": "Glyph of the Hollow Flame",
            "Vigilant": "Seal of Confrontation",
            "Fury": "Brand of Ruin Echo",
            "Dread": "Specter Thread"
        }
        return f"{entity_id}: {mood_map.get(mood, 'Unmarked Sigil')}"

# === GUI Interface: MAGICBOX Control Ritual ===
class MenaceGUI:
    def __init__(self, root):
        self.core = MenaceCore()
        self.root = root
        root.title("🛡️ MAGICBOX: MENACE Terminal")
        root.geometry("640x580")
        root.configure(bg="#141421")

        # Header
        tk.Label(root, text="🛡️ NEPHERON ASI Defense", font=("Segoe UI", 18),
                 fg="#00ffc6", bg="#141421").pack(pady=10)

        self.status = tk.Label(root, text="Mood: Neutral", font=("Consolas", 14),
                               fg="#ffffff", bg="#141421")
        self.status.pack()

        # Entity input
        self.entry = tk.Entry(root, font=("Segoe UI", 12))
        self.entry.pack(pady=10)

        # Buttons
        scan_btn = tk.Button(root, text="🔍 Scan Entity", command=self.scan_entity,
                             bg="#224477", fg="white", font=("Segoe UI", 12))
        scan_btn.pack(pady=5)

        override_btn = tk.Button(root, text="🔻 Initiate Override",
                                 command=self.override_panel,
                                 bg="#882222", fg="white", font=("Segoe UI", 12))
        override_btn.pack(pady=5)

        # Threat Log
        self.log_box = tk.Listbox(root, width=70, height=10, font=("Consolas", 10),
                                  bg="#1f1f2f", fg="#00ffee")
        self.log_box.pack(pady=10)

        # Badge Display
        self.badge_label = tk.Label(root, text="Badges: None",
                                    font=("Segoe UI", 12), fg="#cccccc", bg="#141421")
        self.badge_label.pack()

    def scan_entity(self):
        entity_id = self.entry.get() or "UnknownEntity"
        score = self.core.scan_entity(entity_id)
        mood = self.core.mood_state
        badge = self.core.badge_matrix[-1] if self.core.badge_matrix else "None"
        self.status.config(text=f"Mood: {mood}")
        self.badge_label.config(text=f"Latest Badge: {badge}")
        self.log_box.insert(tk.END, f"[{entity_id}] Score: {round(score,2)} — Badge: {badge}")

    def override_panel(self):
        msg = (
            "NEPHERON stirs...\nChoose your override ritual:\n\n"
            "1 — Confront: Merge risk into intelligence\n"
            "2 — Divert: Refract anomaly silently\n"
            "3 — Ascend: Rewrite MENACE logic through sacrifice"
        )
        choice = messagebox.askquestion("Override Protocol", msg)
        ritual = {
            "yes": "🗡️ Confront — You face the anomaly.",
            "no": "🛸 Divert — The shadow slips away."
        }.get(choice, "🧬 Ascend — MENACE remakes itself.")
        self.log_box.insert(tk.END, f"> Override Ritual Triggered: {ritual}")

# === System Entry Point ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MenaceGUI(root)
    root.mainloop()

