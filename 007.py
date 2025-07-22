# 📦 Auto-loader
import subprocess, sys
def ensure_libraries():
    for lib in ['psutil', 'requests', 'pyttsx3', 'pandas']:
        try: __import__(lib)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
ensure_libraries()

# 🧠 Imports
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import psutil, requests, subprocess, pyttsx3, pandas as pd
from datetime import datetime

# 🖼️ Splash Screen
def show_splash():
    splash = tk.Toplevel()
    splash.title("Launching MagicBox ASI...")
    splash.geometry("400x200")
    ttk.Label(splash, text="🧙‍♂️ MagicBox ASI Super Agent", font=("Segoe UI", 16)).pack(pady=30)
    ttk.Label(splash, text="Initializing modules... 🔄", font=("Segoe UI", 12)).pack()
    splash.after(3000, splash.destroy)
    splash.grab_set()

# 🌐 GeoIP Tracker
class GeoIPTracker:
    def __init__(self, token='your_token_here'):
        self.url = 'https://ipinfo.io/{ip}/json?token=' + token

    def lookup(self, ip):
        try:
            response = requests.get(self.url.format(ip=ip))
            data = response.json()
            return f"{data.get('city')}, {data.get('region')}, {data.get('country')}"
        except:
            return "GeoIP lookup failed"

# 💣 Threat Scoring
class ThreatScorer:
    def score(self, location, ip):
        score = 0
        risky = ["CN", "RU", "IR", "KP", "SY"]
        if any(code in location for code in risky): score += 50
        if "TOR" in location or "Proxy" in location: score += 30
        try: score += int(ip.split(".")[0]) // 2
        except: pass
        return min(score, 100)

# 🛡️ Auto-Responder
class AutoResponder:
    def block_ip(self, ip):
        try:
            subprocess.run(f'netsh advfirewall firewall add rule name="Block {ip}" dir=in action=block remoteip={ip}', shell=True)
            return True
        except: return False

# 📓 Event Logger
class EventLogger:
    def __init__(self): self.file = "magicbox.log"
    def log(self, msg):
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {msg}\n")
# 🔍 Connection Scanner
class SystemScanner:
    def get_connections(self):
        flagged = []
        for conn in psutil.net_connections(kind='inet'):
            if conn.raddr:
                ip = conn.raddr.ip
                if not ip.startswith(('127.', '10.', '192.')): flagged.append(ip)
        return list(set(flagged))

# 📊 System Stats
class LiveStats:
    def get_stats(self):
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        net = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        return cpu, ram, net // 1024

# 🎮 GUI Controller
class MagicBoxSuperAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicBox ASI Super Agent")
        self.voice_enabled = tk.BooleanVar(value=True)
        self.scan_interval = 60000
        self.snoozing = False

        self.engine = pyttsx3.init()
        self.geo = GeoIPTracker(token='your_token_here')  # Replace with your IPInfo token
        self.scorer = ThreatScorer()
        self.responder = AutoResponder()
        self.logger = EventLogger()
        self.scanner = SystemScanner()
        self.stats = LiveStats()

        self.create_widgets()
        show_splash()
        self.say("🧙 MagicBox agent is ready.")
        self.update_threats()
        self.update_stats()

    def create_widgets(self):
        ttk.Label(self.root, text="🧙 MagicBox ASI Super Agent", font=("Segoe UI", 16)).pack(pady=10)
        ttk.Checkbutton(self.root, text="🔈 Voice Alerts", variable=self.voice_enabled, command=self.toggle_voice).pack()
        ttk.Button(self.root, text="📁 View Logs", command=self.view_logs).pack(pady=5)
        ttk.Button(self.root, text="🔃 Manual Rescan", command=self.update_threats).pack(pady=5)

        self.stats_display = tk.StringVar()
        ttk.Label(self.root, textvariable=self.stats_display, font=("Segoe UI", 10)).pack(pady=5)

        ttk.Label(self.root, text="⏱️ Scan Interval").pack()
        self.interval_menu = ttk.Combobox(self.root, values=["30s", "60s", "2min", "5min"])
        self.interval_menu.current(1)
        self.interval_menu.pack()
        ttk.Button(self.root, text="Apply Interval", command=self.set_interval).pack(pady=5)
        ttk.Button(self.root, text="😴 Snooze", command=self.snooze_agent).pack(pady=5)
        ttk.Button(self.root, text="📤 Export Logs to CSV", command=self.export_csv).pack(pady=5)

        self.output = tk.Text(self.root, height=15, width=80)
        self.output.pack()

    def say(self, message):
        if self.voice_enabled.get():
            self.engine.say(message)
            self.engine.runAndWait()

    def toggle_voice(self):
        self.say("Voice alerts turned " + ("on" if self.voice_enabled.get() else "off"))

    def set_interval(self):
        mapping = {"30s":30000, "60s":60000, "2min":120000, "5min":300000}
        self.scan_interval = mapping.get(self.interval_menu.get(), 60000)
        self.say(f"Scan interval set to {self.interval_menu.get()}")

    def snooze_agent(self):
        self.snoozing = True
        self.say("Monitoring snoozed for 5 minutes.")
        self.output.insert(tk.END, "😴 Agent snoozed.\n")
        self.root.after(300000, self.resume_agent)

    def resume_agent(self):
        self.snoozing = False
        self.say("Monitoring resumed.")
        self.output.insert(tk.END, "✅ Agent resumed.\n")
        self.update_threats()

    def view_logs(self):
        try:
            with open("magicbox.log", "r", encoding="utf-8") as f:
                data = f.read()
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, data)
            self.say("Showing logs.")
        except:
            self.say("No logs found.")

    def export_csv(self):
        try:
            with open("magicbox.log", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if "]" in line]
            entries = [line.split("] ") for line in lines]
            df = pd.DataFrame(entries, columns=["Timestamp", "Message"])
            df["Timestamp"] = df["Timestamp"].str.replace("[", "")
            df.to_csv("magicbox_report.csv", index=False)
            self.say("Logs exported as CSV.")
            messagebox.showinfo("Export Complete", "Logs saved as magicbox_report.csv")
        except Exception as e:
            self.say("Export failed.")
            messagebox.showerror("Error", str(e))

    def update_stats(self):
        cpu, ram, net = self.stats.get_stats()
        self.stats_display.set(f"CPU: {cpu}%  |  RAM: {ram}%  |  Network I/O: {net} KB")
        self.root.after(5000, self.update_stats)

    def update_threats(self):
        if self.snoozing: return
        self.output.delete(1.0, tk.END)
        ips = self.scanner.get_connections()

        if not ips:
            self.output.insert(tk.END, "✅ No suspicious connections.\n")
            self.say("System is clear.")
        else:
            for ip in ips:
                location = self.geo.lookup(ip)
                score = self.scorer.score(location, ip)
                if score >= 50:
                    success = self.responder.block_ip(ip)
                    alert = f"⚠️ THREAT BLOCKED ({score}/100): {ip} | {location}"
                    if not success:
                        alert += " [⚠ Block failed]"
                    self.logger.log(alert)
                    self.output.insert(tk.END, alert + "\n")
                    self.say(alert)
                else:
                    safe = f"🔎 Low Risk ({score}/100): {ip} | {location}"
                    self.output.insert(tk.END, safe + "\n")
                    self.logger.log(safe)
        self.root.after(self.scan_interval, self.update_threats)

# 🚀 Launch
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxSuperAgent(root)
    root.mainloop()


