# Auto-install required libraries
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import threading, time, random, sys, os
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])
    import tkinter as tk
    from tkinter import ttk, messagebox
    import threading, time, random

# === MENACE INTELLIGENCE CORE ===
class MenaceCore:
    def __init__(self):
        self.trust_log = {}
        self.badge_matrix = []
        self.memory_anchors = []
        self.mood_state = "Neutral"
        self.threat_score = 0.0
        self.entity_counter = 0

    def auto_scan(self):
        entity_id = f"Entity_{self.entity_counter}"
        self.entity_counter += 1
        score = random.uniform(0, 1)
        self.trust_log[entity_id] = score
        self.memory_anchors.append((entity_id, score))
        self.threat_score = score
        self.update_mood(score)
        if score > 0.75:
            self.trigger_escalation(entity_id)
        return entity_id, score

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
        mood_map = {
            "Clarity": "Glyph of the Hollow Flame",
            "Vigilant": "Seal of Confrontation",
            "Fury": "Brand of Ruin Echo",
            "Dread": "Specter Thread"
        }
        mood = self.mood_state
        return f"{entity_id}: {mood_map.get(mood, 'Unmarked Sigil')}"

# === NEPHERON GUI TERMINAL ===
class MenaceGUI:
    def __init__(self, root):
        self.core = MenaceCore()
        self.root = root
        root.title("🛡️ MAGICBOX: MENACE Autonomous Terminal")
        root.geometry("640x580")
        root.configure(bg="#141421")

        tk.Label(root, text="🛡️ NEPHERON SYSTEM — Autonomous Defense", font=("Segoe UI", 18),
                 fg="#00ffc6", bg="#141421").pack(pady=10)

        self.status = tk.Label(root, text="Mood: Neutral", font=("Consolas", 14),
                               fg="#ffffff", bg="#141421")
        self.status.pack()

        self.badge_label = tk.Label(root, text="Badges: None",
                                    font=("Segoe UI", 12), fg="#cccccc", bg="#141421")
        self.badge_label.pack()

        self.log_box = tk.Listbox(root, width=70, height=12, font=("Consolas", 10),
                                  bg="#1f1f2f", fg="#00ffee")
        self.log_box.pack(pady=10)

        self.start_autoscan()

    def start_autoscan(self):
        def scan_loop():
            while True:
                entity, score = self.core.auto_scan()
                mood = self.core.mood_state
                badge = self.core.badge_matrix[-1] if self.core.badge_matrix else "None"
                log_line = f"[{entity}] Score: {round(score,2)} — Mood: {mood} — Badge: {badge}"
                self.log_box.insert(tk.END, log_line)
                self.status.config(text=f"Mood: {mood}")
                self.badge_label.config(text=f"Latest Badge: {badge}")
                time.sleep(5)
        threading.Thread(target=scan_loop, daemon=True).start()

# === SYSTEM ENTRY POINT ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MenaceGUI(root)
    root.mainloop()

