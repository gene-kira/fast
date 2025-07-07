import os
import sys
import time
import hashlib
import json
import datetime
import platform
import subprocess

# === 🔧 Autoloader: Ensure required libraries are present ===
REQUIRED_LIBS = ["psutil"]
def autoload_libraries():
    print("⟲ Autoloader: Verifying dependencies...")
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
            print(f"🔹 {lib} is present.")
        except ImportError:
            print(f"⚠️ {lib} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
    print("✅ Autoload complete.\n")

autoload_libraries()
import psutil
import sqlite3

DB_PATH = "glyph_store.db"

# === 💾 Database Setup ===
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

# === 🔗 Anchor Hash Generator ===
def generate_anchor_hash(entropy_fingerprint, glyph_sequence):
    timestamp = datetime.datetime.utcnow().isoformat()
    data = entropy_fingerprint + timestamp + ''.join(glyph_sequence)
    return hashlib.sha256(data.encode()).hexdigest()

# === 📜 Memory Imprint Poster ===
def post_glyph_entry(entry):
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

# === 🔁 Rollback Retrieval ===
def retrieve_glyph_entry(anchor_hash):
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

# === 🧪 Entropy Sampler ===
def sample_entropy():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    drivers = "unstable:usb.sys"  # Simulated anomaly
    return {
        "cpu": f"{cpu}%",
        "memory": "fragmented" if memory.percent > 70 else "stable",
        "drivers": drivers
    }

# === 🫀 Heartbeat Loop ===
def start_heartbeat(interval=60):
    print("🫀 SentinelX heartbeat engaged...")
    while True:
        now = datetime.datetime.utcnow().isoformat()
        print(f"⏱️ [{now}] System Pulse")

        entropy = sample_entropy()
        print(f"🌐 Entropy Snapshot: {entropy}")

        # Cast ritual glyphs
        glyphs = ["⟊ :driver :high :patch", "⇌ :kernel :mid :scan"]
        anchor = generate_anchor_hash(json.dumps(entropy), glyphs)
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
                "drivers": entropy["drivers"]
            },
            "trust_vector": 0.91,
            "source_path": "daemon.heartbeat"
        }
        post_glyph_entry(imprint)
        print(f"📜 Glyph Imprinted: {anchor}\n")

        time.sleep(interval)

# === 🔁 Manual Rollback Invocation ===
def rollback_by_anchor(anchor_hash):
    print(f"🔁 Attempting rollback for anchor: {anchor_hash}")
    entry = retrieve_glyph_entry(anchor_hash)
    if entry:
        print(f"🌀 Restoring from glyphs: {entry['glyph_sequence']}")
        print(f"🧬 System snapshot: {entry['system_state']}")
    else:
        print("❌ No matching glyph found.")

# === 🚀 Entry Point ===
if __name__ == "__main__":
    init_db()
    try:
        start_heartbeat(interval=30)  # Set to 30s for fast testing
    except KeyboardInterrupt:
        print("\n🛑 SentinelX daemon halted.")

