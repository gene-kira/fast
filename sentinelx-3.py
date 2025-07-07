import sys, subprocess, time, os, json, hashlib, datetime, platform

# === 📦 Autoloader: Install required packages ===
REQUIRED_LIBS = ["psutil", "pyttsx3", "tkinter"]
def autoload():
    print("🔧 Checking dependencies...")
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
            print(f"✅ {lib} ready")
        except ImportError:
            print(f"📦 Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
autoload()

# === 📦 Imports (after install) ===
import psutil, sqlite3, pyttsx3
import tkinter as tk
from tkinter import ttk

DB_PATH = "glyph_store.db"

# === 🗣️ Voice Engine ===
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        print(f"🔇 Voice unavailable")

def announce_startup():
    speak("SentinelX is online. System watch initiated.")

# === 💾 Database ===
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
    if row:
        return {
            "anchor_hash": row[0],
            "glyph_sequence": json.loads(row[1]),
            "ritual_outcome": row[2],
            "entropy_fingerprint": row[3],
            "timestamp": row[4],
            "system_state": json.loads(row[5]),
            "trust_vector": row[6],
            "source_path": row[7]
        }
    return None

# === 🔗 Anchor hash ===
def generate_hash(entropy, glyphs):
    timestamp = datetime.datetime.utcnow().isoformat()
    raw = entropy + timestamp + ''.join(glyphs)
    return hashlib.sha256(raw.encode()).hexdigest()

# === 🧪 Entropy + Temp Sample ===
def get_entropy():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    drivers = "unstable:usb.sys"
    return {
        "cpu": f"{cpu}%",
        "memory": "fragmented" if mem.percent > 70 else "stable",
        "drivers": drivers
    }

def get_temp():
    temps = {"cpu": "N/A", "gpu": "N/A"}
    try:
        data = psutil.sensors_temperatures()
        key = "coretemp" if "coretemp" in data else list(data.keys())[0]
        if key:
            temps["cpu"] = data[key][0].current
    except: pass
    return temps

# === 🔁 Rollback Invocation ===
def rollback(anchor_hash):
    entry = get_glyph(anchor_hash)
    if entry:
        speak("Rollback sequence initiated.")
        print(f"🌀 Restoring from glyph: {entry['glyph_sequence']}")
        print(f"🧬 Snapshot: {entry['system_state']}")
    else:
        speak("Rollback failed. Glyph not found.")

# === 🫀 Heartbeat loop ===
def heartbeat(interval=30):
    while True:
        now = datetime.datetime.utcnow().isoformat()
        entropy = get_entropy()
        temps = get_temp()

        print(f"\n⏱️ [{now}] Heartbeat")
        print(f"🌐 Entropy: {entropy}")
        print(f"🌡️ CPU Temp: {temps['cpu']}°C")

        if isinstance(temps['cpu'], (int, float)) and temps['cpu'] > 80:
            speak("Warning. CPU temperature is above safe level.")

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
                "cpu_temp": temps["cpu"]
            },
            "trust_vector": 0.91,
            "source_path": "daemon.heartbeat"
        }
        store_glyph(imprint)
        time.sleep(interval)

# === 🖥️ Symbolic UI Scaffold ===
def launch_ui():
    root = tk.Tk()
    root.title("SentinelX Dashboard")
    root.geometry("500x400")

    tk.Label(root, text="🛡️ SentinelX Status", font=("Arial", 16)).pack()

    glyph_frame = tk.Frame(root)
    glyph_frame.pack(pady=10)

    tree = ttk.Treeview(glyph_frame, columns=("anchor", "time"), show="headings")
    tree.heading("anchor", text="Anchor Hash")
    tree.heading("time", text="Timestamp")
    tree.pack()

    def refresh_glyphs():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT anchor_hash, timestamp FROM glyphs ORDER BY timestamp DESC LIMIT 10")
        rows = c.fetchall()
        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", "end", values=(r[0][:8]+"...", r[1]))
        conn.close()

    def on_rollback():
        selected = tree.selection()
        if selected:
            item = tree.item(selected[0])
            anchor = item['values'][0].replace("...", "")
            rollback(anchor)

    tk.Button(root, text="⟳ Refresh Glyphs", command=refresh_glyphs).pack()
    tk.Button(root, text="↺ Trigger Rollback", command=on_rollback).pack()

    refresh_glyphs()
    root.mainloop()

# === 🚀 Start ===
if __name__ == "__main__":
    init_db()
    announce_startup()
    import threading
    threading.Thread(target=heartbeat, daemon=True).start()
    launch_ui()

