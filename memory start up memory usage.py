import os
import sys
import subprocess

# ========== 0. Auto-install Required Packages ==========
required = ['psutil', 'pyttsx3']
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ========== 1. Core Imports ==========
import pickle
import tkinter as tk
from tkinter import ttk
import threading
import psutil
import pyttsx3

# ========== 2. Global Config ==========
MEMORY_FILE = "magicbox_memory.pkl"
PROFILE_FILE = "magicbox_profile.pkl"
DEFAULT_THRESHOLD_MB = 300
REFRESH_RATE_MS = 5000

# ========== 3. Voice Engine Setup ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

# ========== 4. Memory Persistence ==========
def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_data(filename, default={}):
    return pickle.load(open(filename, "rb")) if os.path.exists(filename) else default

# ========== 5. Scan Startup ==========
def scan_startup_folder():
    startup_paths = [
        os.path.join(os.environ['APPDATA'], r'Microsoft\Windows\Start Menu\Programs\Startup'),
        os.path.join(os.environ['ProgramData'], r'Microsoft\Windows\Start Menu\Programs\Startup')
    ]
    entries = []
    for path in startup_paths:
        if os.path.exists(path):
            entries += os.listdir(path)
    return entries

# ========== 6. Memory Tracker ==========
def get_memory_usage(process_name):
    for proc in psutil.process_iter(['name', 'memory_info']):
        if process_name.lower() in proc.info['name'].lower():
            return proc.info['memory_info'].rss // (1024 * 1024)
    return None

def collect_memory_data(programs, threshold):
    mem_data = {}
    for prog in programs:
        mem = get_memory_usage(prog)
        mem_data[prog] = mem
        if mem and mem > threshold:
            speak(f"⚠️ High usage from {prog}: {mem} MB")
    return mem_data

# ========== 7. Emotional Tint ==========
def get_tint_color(avg, threshold):
    return "#ff4c4c" if avg > threshold else "#4ccfff" if avg < 100 else "#e0e0e0"

# ========== 8. GUI ==========
class MagicBoxMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔮 MagicBox Memory Monitor")
        self.root.geometry("620x560")
        self.root.configure(bg="#1c1c1c")
        self.memory_data = {}
        self.programs = scan_startup_folder()
        self.thresholds = load_data(PROFILE_FILE, {"user": DEFAULT_THRESHOLD_MB})
        self.setup_gui()
        self.refresh_memory()

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2e2e2e", fieldbackground="#2e2e2e", foreground="#e0e0e0", rowheight=28, font=("Helvetica", 12))
        style.configure("TLabel", background="#1c1c1c", foreground="#e0e0e0", font=("Papyrus", 16, "italic"))

        ttk.Label(self.root, text="📜 Startup Programs & Memory Usage").pack(pady=10)

        self.tree = ttk.Treeview(self.root, columns=('Program', 'Memory'), show='headings')
        self.tree.heading('Program', text='Program')
        self.tree.heading('Memory', text='Memory (MB)')
        self.tree.pack(expand=True, fill='both', padx=10)

        self.canvas = tk.Canvas(self.root, height=60, bg="#1c1c1c", highlightthickness=0)
        self.canvas.pack(fill='x')
        self.glyph_text = self.canvas.create_text(300, 30, text="🧿 Awaiting Memory Pulse...", fill="#e0e0e0", font=("Helvetica", 14, "bold"))

        # User Profile Threshold Input
        ttk.Label(self.root, text="🔧 Customize High Memory Threshold").pack(pady=5)
        self.threshold_entry = ttk.Entry(self.root)
        self.threshold_entry.insert(0, str(self.thresholds.get("user", DEFAULT_THRESHOLD_MB)))
        self.threshold_entry.pack()
        ttk.Button(self.root, text="💾 Save Threshold", command=self.update_threshold).pack(pady=5)

    def update_threshold(self):
        try:
            new_val = int(self.threshold_entry.get())
            self.thresholds["user"] = new_val
            save_data(self.thresholds, PROFILE_FILE)
            speak(f"New threshold set at {new_val} MB")
        except ValueError:
            speak("Threshold must be a number")

    def refresh_memory(self):
        self.tree.delete(*self.tree.get_children())
        threshold = self.thresholds.get("user", DEFAULT_THRESHOLD_MB)
        self.memory_data = collect_memory_data(self.programs, threshold)
        total = 0
        count = 0
        for prog, mem in self.memory_data.items():
            self.tree.insert('', 'end', values=(prog, mem if mem else "N/A"))
            if mem: total += mem; count += 1
        avg = total // count if count > 0 else 0
        tint = get_tint_color(avg, threshold)
        self.canvas.itemconfig(self.glyph_text, text=f"💫 Avg Memory: {avg} MB", fill=tint)
        self.root.after(REFRESH_RATE_MS, self.refresh_memory)

# ========== 9. Ritual Launcher Stub ==========
def launch_symbolic_ritual():
    print("🕯️ Ritual hook placeholder activated. Awaiting glyph match...")

# ========== 10. Main ==========
if __name__ == "__main__":
    monitor = MagicBoxMonitor()
    monitor.root.mainloop()
    launch_symbolic_ritual()

