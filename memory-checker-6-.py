import subprocess
import sys

# ✅ Auto-install required packages
required = ["psutil", "pyttsx3", "wmi"]
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import tkinter as tk
from tkinter import ttk
import psutil
import pyttsx3
import platform
import threading
import time
import gc

if platform.system() == "Windows":
    import wmi

# 🧠 EMH Diagnostics Core
class EMH:
    def __init__(self):
        self.last_alerts = {}

    def diagnose(self, usage, available):
        if usage > 90 or available < 100 * 1024 * 1024:
            return "critical"
        elif usage > 80:
            return "warning"
        return "stable"

    def intervene(self, level):
        if level == "critical":
            gc.collect()
            return "Memory critical. EMH intervention executed."
        elif level == "warning":
            return "Memory nearing limit. Recommend closing apps."
        return "Memory systems normal."

    def should_alert(self, key):
        now = time.time()
        last = self.last_alerts.get(key, 0)
        if now - last >= 4 * 3600:
            self.last_alerts[key] = now
            return True
        return False

# 🖥️ MagicBox EMH Suite
class MagicBoxEMH:
    def __init__(self, root):
        self.root = root
        self.root.title("🖖 MagicBox EMH Tactical Suite – Starfleet Edition")
        self.root.geometry("980x700")
        self.voice_on = tk.BooleanVar(value=True)
        self.engine = pyttsx3.init()
        self.emh = EMH()
        self.threat_processes = []

        self.red_alert_frame = tk.Frame(self.root, bg="red", height=30)
        self.red_alert_frame.place(x=0, rely=0.0, relwidth=1)
        self.red_alert_frame.lower()

        self.setup_gui()
        self.watchdog_loop()

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#110022")
        style.configure("TLabel", background="#110022", foreground="lightblue", font=("Consolas", 12))
        style.configure("TButton", font=("Consolas", 11), padding=6)
        style.configure("TCheckbutton", background="#110022", foreground="lightblue", font=("Consolas", 11))

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, pady=(35, 0))  # Shift tabs down to avoid red alert bar

        self.tab_mem = ttk.Frame(tabs)
        self.tab_ecc = ttk.Frame(tabs)
        self.tab_process = ttk.Frame(tabs)
        self.tab_override = ttk.Frame(tabs)

        tabs.add(self.tab_mem, text="🧠 Memory")
        tabs.add(self.tab_ecc, text="🔧 ECC")
        tabs.add(self.tab_process, text="👾 Threats")
        tabs.add(self.tab_override, text="🕹️ EMH Override")

        self.label_mem = ttk.Label(self.tab_mem, text="Initializing...")
        self.label_mem.pack(pady=20)

        self.label_ecc = ttk.Label(self.tab_ecc, text="ECC scan loading...")
        self.label_ecc.pack(pady=20)

        self.label_proc = ttk.Label(self.tab_process, text="Scanning...")
        self.label_proc.pack(pady=20)

        self.voice_toggle = ttk.Checkbutton(
            self.tab_override, text="Enable EMH Voice",
            variable=self.voice_on, command=self.toggle_voice
        )
        self.voice_toggle.pack(pady=20)

        ttk.Button(self.tab_override, text="Initiate Repair Sequence", command=self.manual_repair).pack(pady=10)
        ttk.Button(self.tab_override, text="Force Quarantine", command=self.manual_quarantine).pack(pady=10)
        ttk.Button(self.tab_override, text="Core Isolation Protocol", command=self.manual_isolation).pack(pady=10)

    def speak(self, msg, tag="default"):
        if self.voice_on.get() and self.emh.should_alert(tag):
            self.engine.say(msg)
            self.engine.runAndWait()

    def toggle_voice(self):
        state = "enabled" if self.voice_on.get() else "disabled"
        self.speak(f"Voice alerts {state}", tag="voice_toggle")

    def watchdog_loop(self):
        def loop():
            self.update_memory()
            threading.Thread(target=self.check_ecc, daemon=True).start()
            threading.Thread(target=self.scan_processes, daemon=True).start()
            self.root.after(5000, self.watchdog_loop)
        loop()

    def update_memory(self):
        mem = psutil.virtual_memory()
        usage = mem.percent
        available = mem.available
        status = self.emh.diagnose(usage, available)
        response = self.emh.intervene(status)

        display = (
            f"MEMORY STATUS\n"
            f"---------------------\n"
            f"Total: {mem.total // (1024**2)} MB\n"
            f"Used: {mem.used // (1024**2)} MB\n"
            f"Free: {available // (1024**2)} MB\n"
            f"Usage: {usage}%\n\n"
            f"EMH: {response}"
        )

        color = "red" if status == "critical" else "orange" if status == "warning" else "lightblue"
        self.label_mem.config(text=display, foreground=color)

        if status == "critical":
            self.red_alert_frame.lift()
            self.speak("Memory emergency detected. Shields up.", tag="mem_critical")
        else:
            self.red_alert_frame.lower()

    def check_ecc(self):
        if platform.system() != "Windows":
            self.label_ecc.config(text="ECC monitoring not supported on this OS.")
            return
        try:
            c = wmi.WMI()
            ecc_types = {
                1: "Other", 2: "Unknown", 3: "None", 4: "Parity",
                5: "Single-bit ECC", 6: "Multi-bit ECC", 7: "CRC"
            }
            report = ""
            for chip in c.Win32_PhysicalMemory():
                ecc = ecc_types.get(chip.ErrorCorrectionType, "Unknown")
                report += f"{chip.BankLabel}: ECC Type - {ecc}\n"
            self.label_ecc.config(text=report.strip())
            self.speak("ECC status updated.", tag="ecc_ok")
        except Exception:
            self.label_ecc.config(text="ECC check failed.")
            self.speak("ECC system error detected.", tag="ecc_error")

    def scan_processes(self):
        procs = psutil.process_iter(['pid', 'name', 'memory_info'])
        report = ""
        self.threat_processes.clear()
        for proc in procs:
            try:
                mem_used = proc.info['memory_info'].rss // (1024**2)
                report += f"{proc.info['name']} (PID {proc.info['pid']}): {mem_used} MB\n"
                if mem_used > 500:
                    self.threat_processes.append(proc.info)
            except Exception:
                continue
        self.label_proc.config(text=report.strip())
        if self.threat_processes:
            self.red_alert_frame.lift()
            self.speak("Potential rogue process identified.", tag="proc_threat")

    def manual_repair(self):
        gc.collect()
        self.speak("Manual memory repair protocol initiated.", tag="manual_repair")

    def manual_quarantine(self):
        if not self.threat_processes:
            self.speak("No threats detected to quarantine.", tag="no_quarantine")
            return
        try:
            for proc in self.threat_processes:
                psutil.Process(proc['pid']).terminate()
            self.speak("Threat process terminated. System integrity restored.", tag="quarantine")
        except Exception:
            self.speak("Quarantine failed. Manual intervention required.", tag="quarantine_fail")

    def manual_isolation(self):
        gc.collect()
        self.speak("Core isolation sequence complete.", tag="core_isolation")

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxEMH(root)
    root.mainloop()

