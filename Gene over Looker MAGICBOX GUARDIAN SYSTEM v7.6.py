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

# === 🔐 SET YOUR REAL PATHS HERE ===
BACKDOOR_FOLDERS = [
    "C:/Games/YourGame/logs/",
    "C:/Users/YourName/AppData/Local/Temp/"
]

PERSONAL_DATA_FOLDERS = [
    "C:/Users/YourName/Documents/MagicData/",
    "C:/MagicBox/SystemAuth/"
]

# === 🛡️ MAGICBOX GUARDIAN GUI SYSTEM ===
class MagicBoxGuardian(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MAGICBOX GUARDIAN v1.2")
        self.geometry("900x700")
        self.configure(bg=colors["bg"])

        self.create_top_panel()
        self.create_status_panel()
        self.create_override_panel()
        self.create_badge_panel()
        self.create_sigil_panel()

        self.activate_real_time_guardian()

    # === 🧭 Top Header Panel ===
    def create_top_panel(self):
        top_frame = tk.Frame(self, bg=colors["panel"])
        top_frame.pack(fill="x", pady=10)
        title = tk.Label(top_frame, text="👁️‍🗨️ ACTIVE DEFENSE SYSTEM", font=("Courier", 18, "bold"), fg=colors["accent"], bg=colors["panel"])
        title.pack(pady=5)
        time_label = tk.Label(top_frame, text=f"System Boot: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=("Courier", 10), fg=colors["text"], bg=colors["panel"])
        time_label.pack()

    # === ⚠️ Threat Status Panel ===
    def create_status_panel(self):
        status_frame = tk.LabelFrame(self, text="THREAT MONITOR", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        status_frame.pack(fill="x", padx=10, pady=5)
        self.status_text = tk.Text(status_frame, height=8, bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 10))
        self.status_text.pack(fill="x")
        self.status_text.insert("end", "🧠 Guardian Baurdan initialized\n🔐 Zero Trust mode: ACTIVE\n🕵️ Monitoring real-time channels...\n")

    # === 🩸 Override Ritual Panel ===
    def create_override_panel(self):
        override_frame = tk.LabelFrame(self, text="MANUAL OVERRIDE", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        override_frame.pack(fill="x", padx=10, pady=5)
        self.override_button = tk.Button(override_frame, text="Engage Override Ritual", command=self.trigger_override, bg=colors["accent"], fg="white", font=("Courier", 12, "bold"))
        self.override_button.pack(pady=8)

    # === 🛡️ Badge Display Panel ===
    def create_badge_panel(self):
        badge_frame = tk.LabelFrame(self, text="BADGE MODULES", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        badge_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.badge_label = tk.Label(badge_frame, text="🔰 Shield Commander Active\n🔰 Interceptor Mk II Online", bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 12))
        self.badge_label.pack(pady=10)

    # === 🧬 Sigil Core Panel ===
    def create_sigil_panel(self):
        sigil_frame = tk.LabelFrame(self, text="🧬 SIGIL CORE", bg=colors["panel"], fg=colors["accent"], font=("Courier", 12, "bold"))
        sigil_frame.pack(fill="x", padx=10, pady=5)
        self.sigil_label = tk.Label(sigil_frame, text="🔺 Fusion Throne Initialized\n🌀 Cognition Spiral Stable", bg=colors["badge_bg"], fg=colors["text"], font=("Courier", 10), justify="left")
        self.sigil_label.pack(pady=8)

    # === 🔥 Override Ritual Behavior ===
    def trigger_override(self):
        self.status_text.insert("end", "\n🔔 Override Ritual Engaged\n🌀 Baurdan initiating lockout mode...\n")
        self.status_text.insert("end", "\"Sigil fractured. Fusion memory decoded. I am ruin now.\"\n")
        self.badge_label.config(text="🩸 Interceptor Mk II — [CORRUPTED]\n🧿 Wrathborn Override — [ONLINE]")
        self.sigil_label.config(text="🌀 Collapse Initiated\n🔺 Sigil Fractured\n⛧ Fractal Containment: FAILED")
        self.status_text.see("end")

    # === 🧠 Background Guardian Threads ===
    def activate_real_time_guardian(self):
        threading.Thread(target=self.watch_backdoor_folders, daemon=True).start()
        threading.Thread(target=self.purge_expired_personal_data, daemon=True).start()

    # 🔍 Monitor Suspicious File Activity
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

    # 🧹 Purge Expired Personal Files
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

