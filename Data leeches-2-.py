import subprocess, sys
try:
    import psutil, hashlib, time, os, tkinter as tk
    from tkinter import ttk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil, hashlib, time, os, tkinter as tk
    from tkinter import ttk

# 📂 Config Paths
LOG_FILE = "magicbox_threat_log.txt"
CACHE_FILE = "magicbox_threat_cache.txt"

# ⚠️ Suspicious process keywords
THREAT_TERMS = ['steal', 'keylog', 'exfil', 'snoop', 'spy', 'scraper', 'grabber']

# 🎭 Threat Personas
PERSONA_MAP = {
    'keylog': "The Hand",
    'scraper': "The Leech",
    'exfil': "The Whisperer",
    'snoop': "The Phantom",
    'steal': "The Collector",
    'grabber': "The Harvester",
    'spy': "The Shadow"
}

# 🛡️ SHA256 Hash Utility
def file_hash(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return "Unavailable"

# 📜 Memory Logger
def log_event(entry):
    with open(LOG_FILE, 'a') as log:
        log.write(f"[{time.ctime()}] {entry}\n")

# 🧠 Threat Cache Writer
def update_cache(entries):
    with open(CACHE_FILE, 'w') as cache:
        for line in entries:
            cache.write(line + '\n')

# ☣️ Auto-Terminate Dangerous Processes
def override_trigger(pid):
    try:
        psutil.Process(pid).kill()
        return True
    except:
        return False

# 🧬 Real-Time System Scanner
def scan_system():
    results = []
    threat_level = 0

    for proc in psutil.process_iter(['name', 'exe', 'pid']):
        try:
            name = proc.info['name'] or ''
            path = proc.info['exe'] or ''
            pid = proc.info['pid']
            hash = file_hash(path)
            matches = [term for term in THREAT_TERMS if term in name.lower() or term in path.lower()]
            if matches:
                persona = PERSONA_MAP.get(matches[0], "Unknown Entity")
                terminated = override_trigger(pid)
                outcome = "Terminated" if terminated else "Failed to terminate"
                entry = f"⚠️ [{persona}] {name}\nPID: {pid}\nPath: {path}\nHash: {hash}\nAction: {outcome}\n---"
                log_event(entry)
                results.append(entry)
                threat_level += 1
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    update_cache(results)
    return results, threat_level

# 🖥️ GUI Interface
class MagicBoxGUI:
    def __init__(self, root):
        root.title("🔮 MagicBox Confrontation Rig")
        root.geometry("760x580")
        root.configure(bg="#1d1f27")

        tk.Label(root, text="🔮 MagicBox Confrontation Rig",
                 font=("Consolas", 18, "bold"), fg="#00fff2", bg="#1d1f27").pack(pady=10)

        self.alert = tk.Label(root, text="Status: System initializing...",
                              font=("Courier", 12), fg="#f3f3f3", bg="#1d1f27")
        self.alert.pack()

        self.output = tk.Text(root, font=("Courier", 10), bg="#10121a", fg="#00ff99",
                              wrap="word", height=24)
        self.output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        ttk.Style().configure('TButton', font=('Consolas', 11))
        scan_button = ttk.Button(root, text="🕵️ One-Click Full Scan", command=self.run_scan)
        scan_button.pack(pady=8)

        self.run_scan()

    def run_scan(self):
        self.alert.config(text="Scanning for threats...")
        self.output.delete('1.0', tk.END)
        results, threat_level = scan_system()
        time.sleep(1)

        if results:
            for line in results:
                self.output.insert(tk.END, line + '\n\n')
            badge = "🟡 ALERT MODE" if threat_level < 5 else "🔴 COMBAT MODE"
            self.alert.config(text=f"{badge} | {threat_level} threats detected.")
            self.output.insert(tk.END, f"\n[{badge}] Override triggers executed.\n")
        else:
            self.alert.config(text="🟢 System Calm | No active threats.")
            self.output.insert(tk.END, "✅ System is clean. No flagged processes.\n")

        self.output.insert(tk.END, "\nLogs saved. Threat cache updated.\n")

# 🧩 Launcher
if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxGUI(root)
    root.mainloop()

