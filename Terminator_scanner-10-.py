import importlib, os, shutil, threading, tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import time, psutil, pyttsx3, json, hashlib

# 🛠️ AutoLoader
def autoload(mods):
    for m in mods:
        try: importlib.import_module(m)
        except ImportError: messagebox.showerror("Missing", f"Install: {m}")
autoload(["psutil", "pyttsx3", "json", "hashlib"])

# 🔊 Voice Setup
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 0.9)
for v in engine.getProperty("voices"):
    if "Zira" in v.name or "David" in v.name: engine.setProperty("voice", v.id); break
def speak(txt): engine.say(txt); engine.runAndWait()

# 🧠 Memory System
MEMORY_FILE = Path("magicbox_memory.json")
memory = {}
if MEMORY_FILE.exists():
    try: memory = json.load(open(MEMORY_FILE))
    except: memory = {}
def save_memory(): json.dump(memory, open(MEMORY_FILE, "w"))
def get_hash(path): 
    try: return hashlib.md5(open(path, 'rb').read()).hexdigest()
    except: return None

# ⚙️ Config
QUARANTINE = Path.home() / "QuarantineMagicBox"
QUARANTINE.mkdir(exist_ok=True)
DANGER = {"virus", "malware", "hacktool", "crack", "inject"}
SAFE_EXT = {".txt", ".pdf", ".png", ".jpg", ".mp4", ".docx", ".xlsx", ".exe"}
LOG_FILE = Path("magicbox_log.txt")

pending_files, stop_flag, schedule_flag = [], False, False
selected_drive = "C"
unc_path = ""

# 🔍 Scan Logic
def scan_path(path_root):
    global stop_flag, pending_files
    stop_flag = False; pending_files.clear()
    speak(f"Scanning: {path_root}")
    status.set(f"Scanning: {path_root}...")
    log_box.configure(state="normal")
    log_box.insert(tk.END, f"\n🚀 Scanning {path_root}...\n")
    path_root = Path(path_root)

    for dirpath, _, files in os.walk(path_root):
        if stop_flag:
            status.set("Scan aborted."); speak("Scan aborted.")
            log_box.insert(tk.END, "🟥 Scan aborted.\n"); log_box.configure(state="disabled")
            return
        for name in files:
            path = Path(dirpath) / name
            if not path.is_file(): continue
            tag = str(path); ext = path.suffix.lower()
            hashval = get_hash(path); last_hash = memory.get(tag)
            if smart_mode.get() and last_hash == hashval: continue

            flag, reason = False, ""
            if ext not in SAFE_EXT: flag, reason = True, "Uncommon file type"
            for word in DANGER:
                if word in tag.lower(): flag, reason = True, f"Keyword: {word}"
            if path.stat().st_size > 10**8: flag, reason = True, "Suspicious size"
            if flag: pending_files.append((path, reason))

    status.set(f"{len(pending_files)} suspicious items found.")
    speak(f"{len(pending_files)} flagged.")
    log_box.configure(state="disabled")
    review_next()

def review_next():
    if not pending_files:
        status.set("Review complete."); speak("Review complete.")
        return
    path, reason = pending_files.pop(0)
    file_path.set(str(path)); file_reason.set(reason)
    approve_btn.config(state="normal"); quarantine_btn.config(state="normal")

def approve_file():
    memory[file_path.get()] = get_hash(Path(file_path.get())); save_memory()
    log_entry(f"✅ Approved: {file_path.get()} ({file_reason.get()})")
    review_next()

def quarantine_file():
    try:
        shutil.move(file_path.get(), QUARANTINE / Path(file_path.get()).name)
        log_entry(f"🟥 Quarantined: {file_path.get()} ({file_reason.get()})")
        speak(f"Quarantined {Path(file_path.get()).name}")
    except Exception as e: log_entry(f"⚠️ Error: {e}")
    review_next()

def log_entry(entry):
    log_box.configure(state="normal")
    log_box.insert(tk.END, entry + "\n")
    log_box.configure(state="disabled")
    with open(LOG_FILE, "a", encoding="utf-8", errors="ignore") as f: f.write(entry + "\n")

def stop_scan():
    global stop_flag
    stop_flag = True; status.set("Stopping..."); speak("Stopping scan.")

def update_drive(choice):
    global selected_drive
    selected_drive = choice; status.set(f"Drive selected: {selected_drive}")

def scheduled_loop():
    global schedule_flag
    while schedule_flag:
        path_to_scan = unc_path_var.get() or selected_drive + ":\\"
        scan_path(path_to_scan)
        for _ in range(schedule_slider.get()):
            if not schedule_flag: break
            time.sleep(60)

def toggle_schedule():
    global schedule_flag
    schedule_flag = not schedule_flag
    if schedule_flag:
        threading.Thread(target=scheduled_loop, daemon=True).start()
        status.set(f"Auto-scan every {schedule_slider.get()} min."); speak("Scheduled scan active.")
    else:
        status.set("Scheduled scan disabled."); speak("Scheduled scan disabled.")

# 🖥️ GUI Setup
root = tk.Tk()
root.title("🧿 MagicBox • Scheduler & Network Scan")
root.geometry("740x720")
root.configure(bg="#1b1c1e")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 14, "bold"), padding=10)
style.configure("Unified.TButton", foreground="#44bb44", background="#1b1c1e")
style.configure("TLabel", background="#1b1c1e", foreground="#eeeeee", font=("Segoe UI", 12))
style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#ffee60")

frame = ttk.Frame(root)
frame.pack(pady=10)

ttk.Label(frame, text="🧠 MagicBox Sentinel • Scheduler Edition", style="Header.TLabel").pack(pady=8)

# 🖧 UNC Path Input
ttk.Label(frame, text="Optional UNC Path (\\Server\\Share):").pack(pady=4)
unc_path_var = tk.StringVar()
unc_entry = ttk.Entry(frame, textvariable=unc_path_var, width=60)
unc_entry.pack(pady=5)

# 🔽 Drive Selector
ttk.Label(frame, text="Choose Local or Network Drive:").pack()
drive_letters = [p.device[0] for p in psutil.disk_partitions(all=True) if os.path.ismount(p.device)]
selected_drive = drive_letters[0] if drive_letters else "C"
drive_var = tk.StringVar(value=selected_drive)
ttk.OptionMenu(frame, drive_var, selected_drive, *drive_letters, command=update_drive).pack(pady=5)

# ☄️ Buttons
ttk.Button(frame, text="☄️ Scan Path", command=lambda: threading.Thread(target=scan_path, args=(unc_path_var.get() or selected_drive + ":\\",), daemon=True).start(), style="Unified.TButton").pack(pady=5)
ttk.Button(frame, text="⛔ Abort Scan", command=stop_scan, style="Unified.TButton").pack(pady=5)

smart_mode = tk.BooleanVar(value=True)
ttk.Checkbutton(frame, text="🧠 Smart Scan (Skip Reviewed)", variable=smart_mode).pack(pady=5)

# ⏱️ Scheduling System
ttk.Label(frame, text="🕒 Auto Scan Every X Minutes:").pack(pady=5)
schedule_slider = tk.IntVar(value=60)
tk.Scale(frame, from_=5, to=360, orient=tk.HORIZONTAL, variable=schedule_slider).pack()
ttk.Button(frame, text="🔄 Toggle Scheduled Scan", command=toggle_schedule, style="Unified.TButton").pack(pady=5)

# 🔔 Status Display
status = tk.StringVar(value="System ready.")
ttk.Label(frame, textvariable=status, wraplength=600).pack(pady=8)

# 🛡️ Threat Review Panel
ttk.Label(frame, text="Review Threat File:").pack(pady=4)
file_path = tk.StringVar(); file_reason = tk.StringVar()
ttk.Label(frame, textvariable=file_path, wraplength=600).pack()
ttk.Label(frame, textvariable=file_reason).pack(pady=2)

approve_btn = ttk.Button(frame, text="🟩 Approve File", command=approve_file, style="Unified.TButton")
quarantine_btn = ttk.Button(frame, text="🟥 Quarantine File", command=quarantine_file, style="Unified.TButton")
approve_btn.pack(pady=4); quarantine_btn.pack(pady=4)
approve_btn.config(state="disabled"); quarantine_btn.config(state="disabled")

# 📡 Log Console
ttk.Label(frame, text="📡 Scan Log Console").pack(pady=5)
log_box = tk.Text(frame, height=12, width=85, bg="#23262e", fg="#00ffee", font=("Courier New", 10))
log_box.pack()
log_box.insert(tk.END, ">> Awaiting command...\n")
log_box.configure(state="disabled")

# ⚙️ Footer
ttk.Label(frame, text="Powered by MagicBox Fusion Intelligence").pack(pady=10)

root.mainloop()

