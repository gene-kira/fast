import os
import time
import threading
import tkinter as tk
from itertools import cycle
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet

# === Settings === #
TRIGGER_PATH = "C:/guardian_trigger"
VAULT_PATH = "C:/SensitiveData"
HEARTBEAT_RANGE = (50, 100)
key = Fernet.generate_key()
cipher = Fernet(key)

# === Encryption === #
def encrypt_file(path):
    if not os.path.isfile(path): return
    with open(path, 'rb') as f:
        data = f.read()
    enc = cipher.encrypt(data)
    with open(path + ".enc", 'wb') as f:
        f.write(enc)
    update_log(f"🔒 Encrypted: {os.path.basename(path)}")

def decrypt_file(path):
    if not os.path.isfile(path): return
    with open(path, 'rb') as f:
        data = f.read()
    dec = cipher.decrypt(data)
    with open(path.replace(".enc", "_restored.txt"), 'wb') as f:
        f.write(dec)
    update_log(f"🔓 Decrypted: {os.path.basename(path)}")

# === Threat Monitor === #
class GuardianMonitor(FileSystemEventHandler):
    def on_created(self, event):
        update_log(f"🚨 Threat: {event.src_path}")
        update_status("lockdown")
        trigger_lockdown()

def start_monitoring():
    if not os.path.exists(VAULT_PATH): os.makedirs(VAULT_PATH)
    observer = Observer()
    observer.schedule(GuardianMonitor(), VAULT_PATH, recursive=True)
    observer.start()
    update_log(f"👁️ Monitoring: {VAULT_PATH}")
    update_status("monitoring")

# === Lockdown Logic === #
def trigger_lockdown():
    update_log("🛑 Lockdown: Encrypting vault...")
    for f in os.listdir(VAULT_PATH):
        full_path = os.path.join(VAULT_PATH, f)
        if os.path.isfile(full_path):
            encrypt_file(full_path)

# === Validation === #
def detect_trigger(): return os.path.exists(TRIGGER_PATH)
def read_pulse(): return 72
def validate_heartbeat(): return HEARTBEAT_RANGE[0] < read_pulse() < HEARTBEAT_RANGE[1]

# === Status Badge === #
def badge_animation(status):
    palette = {
        "safe": "#00ff88", "monitoring": "#ffff33", "lockdown": "#ff0033"
    }
    colors = palette.get(status, "#ffffff")
    glyphs = cycle(["●", "◍", "◎", "◉"])
    def pulse():
        badge.set(next(glyphs))
        badge_label.config(fg=colors)
        root.after(250, pulse)
    pulse()

def update_status(s):
    update_log(f"📛 Status: {s.upper()}")
    badge_animation(s)

# === Boot Sequences === #
def stealth_boot():
    update_log("🕶️ Stealth Boot...")
    for i in range(5, 0, -1):
        update_log(f"🎬 T-minus {i}")
        time.sleep(1)
    update_log("💨 NeoTokyo shimmer\n🛡️ Guardian Armed")
    update_status("safe")

def predator_boot():
    glyphs = ["⌖", "✪", "⦿", "⛶", "⫸"]
    update_log("🔫 Predator Mode...")
    for i in range(5, 0, -1):
        update_log(f"{glyphs[i % len(glyphs)]} Locking... {i}")
        time.sleep(0.7)
    update_log("☠️ Target Acquired\n🛡️ Guardian Armed")
    update_status("safe")

# === Full Launch Flow === #
def activate_guardian(mode):
    update_log("🔐 Verifying access...")
    if not detect_trigger():
        update_log("⚠️ Trigger missing (bypassed)")
    if not validate_heartbeat():
        update_log("⚠️ Heartbeat failed (bypassed)")
    update_log("✅ Access granted. Initiating boot...")
    time.sleep(1)
    if mode == "stealth": stealth_boot()
    elif mode == "predator": predator_boot()
    start_monitoring()

# === GUI Setup === #
def update_log(msg):
    log.set(msg)
    log_label.update()

def launch_gui(mode):
    btn_frame.pack_forget()
    threading.Thread(target=lambda: activate_guardian(mode)).start()

root = tk.Tk()
root.title("🧿 MagicBox Guardian Console")
root.geometry("440x320")
root.configure(bg="#0f0f0f")

# Title
title = tk.Label(root, text="MAGICBOX GUARDIAN", font=("Consolas", 16), fg="#00ffe0", bg="#0f0f0f")
title.pack(pady=8)

# Badge
badge = tk.StringVar()
badge.set("●")
badge_label = tk.Label(root, textvariable=badge, font=("Consolas", 32), bg="#0f0f0f")
badge_label.pack(pady=4)

# Log Console
log = tk.StringVar()
log.set("Select your boot mode below...")
log_label = tk.Label(root, textvariable=log, font=("Consolas", 12), fg="#ffffff", bg="#0f0f0f", wraplength=400, justify="left")
log_label.pack(pady=8)

# Buttons
btn_frame = tk.Frame(root, bg="#0f0f0f")
btn_frame.pack()

btn1 = tk.Button(btn_frame, text="🎬 Stealth Mode", command=lambda: launch_gui("stealth"),
    font=("Consolas", 12), bg="#222", fg="#00ff88", width=18)
btn2 = tk.Button(btn_frame, text="🔫 Predator Mode", command=lambda: launch_gui("predator"),
    font=("Consolas", 12), bg="#222", fg="#ff0055", width=18)

btn1.grid(row=0, column=0, padx=10, pady=5)
btn2.grid(row=0, column=1, padx=10, pady=5)

root.mainloop()

