import importlib
import tkinter as tk
from tkinter import messagebox
import psutil
import random
from datetime import datetime

# 🛠 Autoloader for required libraries
def load_libraries():
    required_libs = ['tkinter', 'psutil', 'random', 'datetime']
    for lib in required_libs:
        try:
            importlib.import_module(lib)
        except ImportError:
            messagebox.showerror("Missing Library", f"{lib} is required!")
            exit()

# 🧠 FauxThread class for simulated personalities
class FauxThread:
    def __init__(self, name, mood, intensity):
        self.name = name
        self.mood = mood
        self.intensity = intensity

    def get_status(self):
        return f"{self.name} [{self.mood}] — Intensity: {self.intensity}%"

    def execute(self):
        print(f"[FauxCPU] Executing {self.name} ({self.mood}) @ {self.intensity}%")

# 🎩 Generate sample threads
def generate_threads():
    moods = ["Snark", "Wise", "Playful", "Melancholy"]
    return [FauxThread(f"Thread-{i}", random.choice(moods), random.randint(10, 100)) for i in range(4)]

# 🧠 Refresh ThreadPulse Dashboard
def refresh_dashboard():
    threads = generate_threads()
    canvas.delete("all")
    y = 20
    for thread in threads:
        status = thread.get_status()
        bar_length = thread.intensity * 3
        color = "#ffb74d" if thread.mood == "Snark" else "#64b5f6"
        canvas.create_text(20, y, anchor="w", font=("Arial", 10), fill="white", text=status)
        canvas.create_rectangle(200, y - 8, 200 + bar_length, y + 8, fill=color, outline="")
        if thread.intensity >= 90:
            canvas.create_text(20, y + 15, anchor="w", font=("Arial", 9), fill="#ff5252",
                               text=f"⚠ {thread.mood} overload! Echo spawn imminent.")
        y += 40

# 💽 Drive selector
def get_available_drives():
    partitions = psutil.disk_partitions()
    return [p.device for p in partitions if 'rw' in p.opts]

def select_drive():
    selected = drive_var.get()
    if selected:
        with open("neuro_mount.cfg", "w") as f:
            f.write(f"Selected Brain Drive: {selected}")
        messagebox.showinfo("Drive Mounted", f"{selected} dedicated to Butler Suite.")
        status_label.config(text=f"🧠 Status: {selected} mounted as brain drive")
    else:
        messagebox.showerror("Error", "Please select a drive.")

def rescan_drives():
    drive_menu_frame.pack_forget()
    setup_drive_selector()

def setup_drive_selector():
    global drive_var, drive_menu_frame
    drives = get_available_drives()
    drive_menu_frame = tk.Frame(root, bg="#2e2e2e")
    drive_menu_frame.pack()

    drive_var = tk.StringVar(value=drives[0] if drives else "")
    for d in drives:
        tk.Radiobutton(drive_menu_frame, text=d, variable=drive_var, value=d,
                       font=("Arial", 10), bg="#424242", fg="white", selectcolor="#616161").pack(anchor="w", padx=30)
    tk.Button(drive_menu_frame, text="⚙️ Mount & Activate", command=select_drive,
              font=("Arial", 10), bg="#6d4c41", fg="white").pack(pady=5)
    tk.Button(drive_menu_frame, text="🔄 Rescan Drives", command=rescan_drives,
              font=("Arial", 10), bg="#455a64", fg="white").pack(pady=5)

# 🎨 GUI Initialization
def launch_magicbox():
    global canvas, status_label, root
    root = tk.Tk()
    root.title("✨ MagicBox | Butler Personality Suite")
    root.geometry("520x560")
    root.configure(bg="#2e2e2e")

    tk.Label(root, text="🌟 Butler Personality Suite 🌟", font=("Arial", 14), fg="#fdd835", bg="#2e2e2e").pack(pady=10)

    # 💾 Memory Chamber - Drive Selector
    tk.Label(root, text="📦 Memory Chamber", font=("Arial", 12), fg="#ffe082", bg="#2e2e2e").pack()
    setup_drive_selector()

    # 🧵 ThreadPulse Dashboard
    tk.Label(root, text="🧬 FauxThread Dashboard", font=("Arial", 12), fg="#81d4fa", bg="#2e2e2e").pack(pady=5)
    canvas = tk.Canvas(root, width=480, height=180, bg="#37474f")
    canvas.pack()

    tk.Button(root, text="🔁 Refresh Threads", command=refresh_dashboard,
              font=("Arial", 10), bg="#607d8b", fg="white").pack(pady=10)

    # 🧠 Status Label
    status_label = tk.Label(root, text="🧠 Status: Idle", font=("Arial", 10), fg="#c5e1a5", bg="#2e2e2e")
    status_label.pack(side="bottom", pady=10)

    refresh_dashboard()
    root.mainloop()

# 🚀 Launch Butler MagicBox Suite
if __name__ == "__main__":
    load_libraries()
    launch_magicbox()

