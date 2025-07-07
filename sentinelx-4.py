import sys, subprocess, time, os, json, hashlib, datetime, platform, threading

# === 📦 Autoloader for required libraries ===
REQUIRED_LIBS = ["psutil", "pyttsx3"]
def autoload_libraries():
    print("🔧 Autoloading libraries...")
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
            print(f"✅ {lib} loaded.")
        except ImportError:
            print(f"📦 Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
autoload_libraries()

# === Imports ===
import psutil, pyttsx3
import sqlite3
import tkinter as tk
from tkinter import ttk

unit_state = {"temp_unit": "C"}  # 🌡️ Default Celsius
DB_PATH = "glyph_store.db"

# === 🗣️ Voice Engine ===
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        print("🔇 Voice error")

def announce_startup():
    speak("SentinelX is online. System watch initiated.")

# === 💾 Glyph Database ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS glyphs (
            anchor_hash TEXT PRIMARY KEY,
            glyph_sequence TEXT,
            ritual_outcome TEXT,
            entropy_fingerprint TEXT,
            timestamp TEXT,
            system_state TEXT,
            trust_vector REAL,
            source_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_glyph(entry):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO glyphs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        entry['anchor_hash'],
        json.dumps(entry['glyph_sequence']),
        entry['ritual_outcome'],
        entry['entropy_fingerprint'],
        entry['timestamp'],
        json.dumps(entry['system_state']),
        entry['trust_vector'],
        entry['source_path']
    ))
    conn.commit()
    conn.close()

def get_glyph(anchor_hash):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM glyphs WHERE anchor_hash = ?', (anchor_hash,))
    row = c.fetchone()
    conn.close()
    return {
        "anchor_hash": row[0],
        "glyph_sequence": json.loads(row[1]),
        "ritual_outcome": row[2],
        "entropy_fingerprint": row[3],
        "timestamp": row[4],
        "system_state": json.loads(row[5]),
        "trust_vector": row[6],
        "source_path": row[7]
    } if row else None

# === 🔗 Hash Generator ===
def generate_hash(entropy, glyphs):
    raw = entropy + datetime.datetime.utcnow().isoformat() + ''.join(glyphs)
    return hashlib.sha256(raw.encode()).hexdigest()

# === 🧪 Entropy, Temperature, Disk ===
def convert_temp(celsius):
    if unit_state["temp_unit"] == "F":
        return round((celsius * 9 / 5) + 32, 1)
    return round(celsius, 1)

def get_temp():
    temps = {"cpu": "N/A", "gpu": "N/A"}
    try:
        sensor = psutil.sensors_temperatures()
        if sensor:
            key = list(sensor.keys())[0]
            temps["cpu"] = convert_temp(sensor[key][0].current)
    except:
        pass
    return temps

def get_entropy():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    return {
        "cpu": f"{cpu}%",
        "memory": "fragmented" if mem.percent > 70 else "stable",
        "drivers": "unstable:usb.sys"
    }

def sample_disk_health():
    disk = psutil.disk_usage('/')
    return {
        "total": f"{disk.total // (1024**3)}GB",
        "used": f"{disk.used // (1024**3)}GB",
        "free": f"{disk.free // (1024**3)}GB",
        "percent": f"{disk.percent}%"
    }

def get_drive_temp():
    try:
        output = subprocess.run(["smartctl", "-A", "/dev/sda"], capture_output=True, text=True)
        for line in output.stdout.splitlines():
            if "Temperature_Celsius" in line or "Drive Temperature" in line:
                return line.strip()
    except Exception as e:
        return f"SMART error: {e}"
    return "Unavailable"

# === 🔁 Rollback Ritual ===
def rollback(anchor_hash):
    entry = get_glyph(anchor_hash)
    if entry:
        speak("Rollback sequence initiated.")
        print(f"↺ Glyph: {entry['glyph_sequence']}")
        print(f"🧬 Snapshot: {entry['system_state']}")
    else:
        speak("Rollback failed. No glyph found.")

# === 🫀 Heartbeat ===
def heartbeat(interval=30):
    while True:
        now = datetime.datetime.utcnow().isoformat()
        entropy = get_entropy()
        temps = get_temp()
        disk = sample_disk_health()
        drive = get_drive_temp()

        print(f"\n⏱️ [{now}] Heartbeat")
        print(f"🌐 Entropy: {entropy}")
        print(f"🌡️ CPU Temp: {temps['cpu']}°{unit_state['temp_unit']}")
        print(f"💾 Disk: {disk}")
        print(f"📊 Drive Temp: {drive}")

        if isinstance(temps["cpu"], (int, float)) and temps["cpu"] > 80:
            speak(f"Warning. CPU temperature is {temps['cpu']} degrees {unit_state['temp_unit']}.")

        glyphs = ["⟊ :driver :high :patch", "⇌ :kernel :mid :scan"]
        anchor = generate_hash(json.dumps(entropy), glyphs)
        imprint = {
            "anchor_hash": anchor,
            "glyph_sequence": glyphs,
            "ritual_outcome": "partial-success",
            "entropy_fingerprint": json.dumps(entropy),
            "timestamp": now,
            "system_state": {
                "os": platform.system(),
                "cpu": entropy["cpu"],
                "memory": entropy["memory"],
                "drivers": entropy["drivers"],
                "cpu_temp": f"{temps['cpu']}°{unit_state['temp_unit']}",
                "disk": disk,
                "drive_temp": drive
            },
            "trust_vector": 0.91,
            "source_path": "daemon.heartbeat"
        }
        store_glyph(imprint)
        time.sleep(interval)

# === 🖥️ Symbolic UI Dashboard ===
def launch_ui():
    root = tk.Tk()
    root.title("SentinelX Dashboard")
    root.geometry("600x500")

    tk.Label(root, text="🛡️ SentinelX System Monitor", font=("Arial", 16)).pack()
    frame = tk.Frame(root)
    frame.pack(pady=10)

    tree = ttk.Treeview(frame, columns=("anchor", "timestamp"), show="headings")
    tree.heading("anchor", text="Anchor")
    tree.heading("timestamp", text="Time")
    tree.pack()

    def refresh():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT anchor_hash, timestamp FROM glyphs ORDER BY timestamp DESC LIMIT 10")
        rows = cur.fetchall()
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", "end", values=(row[0][:10] + "...", row[1]))
        conn.close()

    def invoke_rollback():
        selected = tree.selection()
        if selected:
            anchor = tree.item(selected[0])['values'][0].replace("...", "")
            rollback(anchor)

    def toggle_units():
        unit_state["temp_unit"] = "F" if unit_state["temp_unit"] == "C" else "C"
        speak(f"Units set to {unit_state['temp_unit']}")
        refresh()

    tk.Button(root, text="🔃 Refresh Logs", command=refresh).pack()
    tk.Button(root, text="↺ Rollback Glyph", command=invoke_rollback).pack()
    tk.Button(root, text="🌡️ Toggle °C/°F", command=toggle_units).pack()

    refresh()
    root.mainloop()

# === 🚀 Main Entry ===
if __name__ == "__main__":
    init_db()
    announce_startup()
    threading.Thread(target=heartbeat, daemon=True).start()
    launch_ui()

