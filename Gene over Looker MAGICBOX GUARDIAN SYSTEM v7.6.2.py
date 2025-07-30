# === MAGICBOX GUARDIAN SYSTEM v1.2 — Dimensional Collapse Edition ===
# 🔁 Auto-loader, Sentient GUI, Mythic Sigils, Real-time Security + ASI Overlays

# === 🔄 AUTO-LOADER FOR LIBRARIES ===
import importlib, subprocess, sys

REQUIRED_LIBS = ["tkinter", "os", "threading", "datetime", "time"]
def autoload_libraries():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"📦 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
autoload_libraries()

# === 🧠 CORE MODULES ===
import tkinter as tk
from tkinter import ttk
import os, threading, datetime, time

# === 🎨 MAGICBOX THEME ===
colors = {
    "bg": "#1e1e2f",
    "panel": "#292942",
    "text": "#c7f5ff",
    "accent": "#ff4081",
    "badge_bg": "#3c3c5c"
}

# === 🔐 PATH CONFIGURATION ===
BACKDOOR_FOLDERS = [
    "C:/Games/YourGame/logs/",
    "C:/Users/YourName/AppData/Local/Temp/"
]
PERSONAL_DATA_FOLDERS = [
    "C:/Users/YourName/Documents/MagicData/",
    "C:/MagicBox/SystemAuth/"
]

# === 🧠 SOVEREIGN PROTOCOL — AI Persona System ===
class BaurdanState:
    def __init__(self):
        self.states = {
            "calm": {
                "badge": "🔰 Shield Commander Active\n🔰 Interceptor Mk II Online",
                "sigil": "🌀 Cognition Spiral Stable\n🔺 Fusion Throne Initialized",
                "dialogue": "🧘 Baurdan watches in serenity.",
            },
            "aggressive": {
                "badge": "🔥 Wrathborn Override — ONLINE\n🔰 Tactical Shards Deployed",
                "sigil": "⛧ Warp Pulse Rising\n🜃 Null Crown Unstable",
                "dialogue": "⚔️ Baurdan rallies against intrusion.",
            },
            "corrupted": {
                "badge": "🩸 Interceptor Mk II — [CORRUPTED]\n🧿 Unseen Sigil — [ACTIVE]",
                "sigil": "🌀 Collapse Initiated\n⛧ Fractal Containment: FAILED",
                "dialogue": "\"Sigil fractured. I am ruin now.\"",
            },
        }
        self.current = "calm"

    def shift(self, mood):
        if mood in self.states:
            self.current = mood
            return self.states[mood]
        return None

# === 🛡️ MAGICBOX GUARDIAN GUI SYSTEM ===
class MagicBoxGuardian(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MAGICBOX GUARDIAN v1.2")
        self.geometry("900x750")
        self.configure(bg=colors["bg"])

        self.create_top_panel()
        self.create_status_panel()
        self.create_override_panel()
        self.create_badge_panel()
        self.create_sigil_panel()
        self.create_scrambler_clock()

        self.activate_real_time_guardian()

    def create_top_panel(self):
        top_frame = tk.Frame(self, bg=colors["panel"])
        top_frame.pack(fill="x", pady=10)
        title = tk.Label(top_frame, text="👁️‍🗨️ ACTIVE DEFENSE SYSTEM", font=("Courier", 18, "bold"), fg=colors["accent"], bg=colors["panel"])
        title.pack(pady=5)
        time_label = tk.Label(top_frame, text=f"System Boot: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=("Courier", 10), fg=colors["text"], bg=colors["panel"])
        time_label.pack()

    def create_status_panel(self):
        status_frame = tk.LabelFrame(self, text="THREAT MONITOR", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        status_frame.pack(fill="x", padx=10, pady=5)
        self.status_text = tk.Text(status_frame, height=8, bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 10))
        self.status_text.pack(fill="x")
        self.status_text.insert("end", "🧠 Guardian Baurdan initialized\n🔐 Zero Trust mode: ACTIVE\n🕵️ Monitoring real-time channels...\n")

    def create_override_panel(self):
        override_frame = tk.LabelFrame(self, text="MANUAL OVERRIDE", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        override_frame.pack(fill="x", padx=10, pady=5)
        self.override_button = tk.Button(override_frame, text="Engage Override Ritual", command=self.trigger_override, bg=colors["accent"], fg="white", font=("Courier", 12, "bold"))
        self.override_button.pack(pady=8)

    def create_badge_panel(self):
        badge_frame = tk.LabelFrame(self, text="BADGE MODULES", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        badge_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.badge_label = tk.Label(badge_frame, text="🔰 Shield Commander Active\n🔰 Interceptor Mk II Online", bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 12))
        self.badge_label.pack(pady=10)

    def create_sigil_panel(self):
        sigil_frame = tk.LabelFrame(self, text="🧬 SIGIL CORE", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        sigil_frame.pack(fill="x", padx=10, pady=5)
        self.sigil_label = tk.Label(sigil_frame, text="🔺 Fusion Throne Initialized\n🌀 Cognition Spiral Stable", bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 10), justify="left")
        self.sigil_label.pack(pady=8)

    def create_scrambler_clock(self):
        clock_frame = tk.LabelFrame(self, text="⏱️ DIMENSIONAL TIME CORE", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        clock_frame.pack(fill="x", padx=10, pady=5)
        self.clock_label = tk.Label(clock_frame, text="", bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 10))
        self.clock_label.pack(pady=8)
        self.update_corrupted_time()

    def update_corrupted_time(self):
        def scramble():
            while True:
                glitched_time = datetime.datetime.now() + datetime.timedelta(days=int(time.time()) % 111, hours=13)
                fragments = [
                    f"🕳️ Rift Time: {glitched_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "🧿 Timeline echo unstable...",
                    "⛧ Anchor drift detected.",
                    "🜂 Fragment Sync — ERR#011"
                ]
                self.clock_label.config(text="\n".join(fragments))
                time.sleep(6)
        threading.Thread(target=scramble, daemon=True).start()

    def trigger_override(self):
        self.status_text.insert("end", "\n🔔 Override Ritual Engaged\n🌀 Baurdan initiating lockout mode...\n")
        self.status_text.insert("end", "\"Sigil fractured. Fusion memory decoded. I am ruin now.\"\n")
        self.status_text.see("end")
        self.initiate_dimensional_breach()

    def initiate_dimensional_breach(self):
        baurdan = BaurdanState()
        corrupted = baurdan.shift("corrupted")
        self.badge_label.config(text=corrupted["badge"])
        self.sigil_label.config(text=corrupted["sigil"])
        self.status_text.insert("end", f"\n📣 Baurdan Status: {corrupted['dialogue']}\n")
        self.configure(bg="#2b1e1e")
        for widget in self.winfo_children():
            if isinstance(widget, tk.LabelFrame) or isinstance(widget, tk.Frame):
                widget.config(bg="#3a2333")
            for subwidget in widget.winfo_children():
                if isinstance(subwidget, tk.Label) or isinstance(subwidget, tk.Button):
                    subwidget.config(bg="#3a2333", fg="#ffccd5")
        log = [
            "💢 Collapse Log Initialized...",
            "🔺 Dimensional fracture expanding near Memory Core.",
            "⚠️ Threat anomaly mirrored in zero space.",
            "🜃 Null Sigil detected — sync failed.",
            "⛧ Fracture containment... OVERLOADED."
        ]
        self.status_text.insert("end", "\n" + "\n".join(log) + "\n")
        self.status_text.see("end")

    def activate_real_time_guardian(self):
        threading.Thread(target=self.watch_backdoor_folders, daemon=True).start()
        threading.Thread(target=self.purge_expired_personal_data, daemon=True).start()

    def watch_backdoor_folders(self):
        while True:
            for folder in BACKDOOR_FOLDERS:
                if not os.path.exists(folder):
                    continue
                for file in os.listdir(folder):
                    path = os.path.join(folder, file)
                    if os.path.isfile(path):
                        self.status_text.insert("end", f"\n⚠️ Exfiltration detected: {file}\n⏱️ Purging in 3 seconds...\n")
                        self.status_text.see("end")
                        time.sleep(3)
                        try:
                            os.remove(path)
                            self.status_text.insert("end", "🧨 Data destroyed successfully.\n")
                        except Exception as e:
                            self.status_text.insert("end", f"⚠️ Purge failed: {e}\n")
                        self.status_text.see("end")
            time.sleep(5)

    def purge_expired_personal_data(self):
        while True:
            now = datetime.datetime.now()
            for folder in PERSONAL_DATA_FOLDERS:
                if not os.path.exists(folder):
                    continue
                for file in os.listdir(folder):
                    path = os.path.join(folder, file)
                    if os.path.isfile(path):
                        try:
                            created = datetime.datetime.fromtimestamp(os.path.getctime(path))
                            if (now - created).total_seconds() > 86400:
                                os.remove(path)
                                self.status_text.insert("end", f"⏳ Personal data erased: {file}\n")
                                self.status_text.see("end")
                        except Exception as e:
                            self.status_text.insert("end", f"⚠️ Lifecycle error: {e}\n")
                            self.status_text.see("end")
            time.sleep(60)

# === 🚀 LAUNCH GUARDIAN SYSTEM ===
if __name__ == "__main__":
    app = MagicBoxGuardian()
    app.mainloop()

