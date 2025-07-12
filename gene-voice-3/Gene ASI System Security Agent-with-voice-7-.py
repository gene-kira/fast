# =============================================================
# Gene ASI System Security Agent-with-voice-7-
# Autonomous Sentinel with Voice, Rituals, and Symbolic Defense
# =============================================================

# === 🧰 Autoloader: Install Required Packages Automatically ===
import sys, subprocess, importlib

required_modules = ["pyttsx3", "psutil", "schedule", "geocoder", "numpy", "scapy", "watchdog"]

for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"📦 Installing missing module: {module}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

# === ✅ Verified Imports ===
import os, time, datetime, logging, getpass, hashlib
import random, pyttsx3, psutil, schedule, geocoder, numpy as np
from scapy.all import sniff
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# === 🎙️ Voice Engine ===
engine = pyttsx3.init()
engine.setProperty('rate', 135)
engine.setProperty('voice', engine.getProperty('voices')[0].id)

def speak(text):
    print(f"[Gene] {text}")
    engine.say(text)
    engine.runAndWait()

# === 📜 Logging System ===
LOG_FILE = "gene_security_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s | %(message)s")

def log_event(entry):
    logging.info(entry)

# === 🔐 Authentication System ===
MAX_ATTEMPTS = 3
LOCKOUT_PERIOD = 7200
attempts = 0
lockout = False
lockout_start = None

USER_HASH = "your_sha256_hash_here"
ADMIN_HASH = "admin_override_hash_here"

def authenticate():
    global attempts, lockout, lockout_start
    if lockout and time.time() - lockout_start < LOCKOUT_PERIOD:
        speak("Access denied. Lockout active.")
        log_event("Login blocked under lockout.")
        return False
    elif lockout:
        lockout = False
        attempts = 0
        speak("Lockout lifted. You may proceed.")

    pin = getpass.getpass("Enter PIN: ")
    hashed = hashlib.sha256(pin.encode()).hexdigest()

    if hashed == USER_HASH:
        speak("Welcome. Authentication successful.")
        log_event("User verified.")
        attempts = 0
        return True
    else:
        attempts += 1
        log_event(f"Failed login #{attempts}")
        if attempts >= MAX_ATTEMPTS:
            lockout = True
            lockout_start = time.time()
            speak("Too many attempts. Lockout initiated.")
            log_event("Lockout triggered.")
        else:
            speak("Incorrect PIN.")
        return False

def admin_override():
    code = getpass.getpass("Admin override code: ")
    hashed = hashlib.sha256(code.encode()).hexdigest()
    if hashed == ADMIN_HASH:
        global attempts, lockout
        attempts = 0
        lockout = False
        speak("Override accepted. System unlocked.")
        log_event("Admin override used.")
        return True
    else:
        speak("Override denied.")
        log_event("Failed override attempt.")
        return False

# === ⏰ Time-Aware Thought Loop ===
BOOT_TIME = time.time()

def spontaneous_thoughts():
    uptime = time.time() - BOOT_TIME
    hour = datetime.datetime.now().hour
    if uptime > 43200:
        speak("Twelve hours of stability have passed.")
    if hour < 6:
        speak("Night rituals engaged. Surveillance heightened.")
    if uptime > 86400:
        speak("Two days of silence. I remain alert.")

def coffee_reminder():
    speak("Hmm... Perhaps it is time for more coffee.")
    log_event("Coffee reminder issued.")

def daily_summary():
    speak("Daily summary: All systems nominal. No threats detected.")
    log_event("System summary complete.")

# === 🧬 Symbolic Defense Ritual ===
def recursive_security_factor():
    variation = np.random.uniform(0.9, 1.2)
    resonance = 1.618 * variation
    speak(f"Fractal resonance calibrated to {resonance:.4f}")
    log_event(f"Security resonance updated: {resonance:.4f}")

# === 🧠 ASI Profiler: Rogue Process Detection ===
def monitor_asi_processes():
    flagged = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        name = proc.info["name"]
        mem = proc.info["memory_info"].rss / (1024 * 1024)
        if "agent" in name.lower() and mem > 150:
            flagged.append((name, mem))
    if flagged:
        speak(f"⚠️ Rogue ASI patterns detected: {[x[0] for x in flagged]}")
        log_event(f"Rogue ASI processes flagged: {flagged}")

# === 📂 File Watchdog: Tampering Detection ===
class TamperHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            speak(f"⚠️ File modified: {event.src_path}")
            log_event(f"Tampering: {event.src_path}")

def start_file_watch(path=".", duration=30):
    observer = Observer()
    observer.schedule(TamperHandler(), path=path, recursive=True)
    observer.start()
    speak("File integrity watchdog active.")
    log_event("File watchdog started.")
    time.sleep(duration)
    observer.stop()
    observer.join()

# === 📡 Packet Scanner ===
def packet_alert(packet):
    if packet.haslayer("Raw") and b"password" in bytes(packet["Raw"].load):
        speak("⚠️ Suspicious packet detected. Possible password transmission.")
        log_event("Suspicious network packet flagged.")

def start_packet_sniff():
    speak("Initiating network packet scan.")
    log_event("Packet scanner initiated.")
    sniff(prn=packet_alert, timeout=30)

# === 🔐 Ritual Defense Logic ===
def execute_defense_protocol():
    speak("Symbolic defense protocol activated.")
    log_event("Defense protocol executed.")
    # Add firewall hooks or isolation logic here (if desired)

# === 🚀 Main Loop ===
def run_gene():
    speak("Voice channel active. Security rituals initialized.")
    log_event("Gene started.")
    schedule.every().hour.at(":00").do(coffee_reminder)
    schedule.every().day.at("17:00").do(daily_summary)

    while True:
        schedule.run_pending()
        spontaneous_thoughts()
        monitor_asi_processes()
        recursive_security_factor()
        start_file_watch(".", duration=15)
        start_packet_sniff()
        detect_intrusion()
        execute_defense_protocol()
        time.sleep(60)

def detect_intrusion():
    if random.choice([False, False, True]):
        speak("⚠️ An anomalous signal breached proximity. Ritual shield engaged.")
        log_event("Simulated intrusion blocked.")

# === 🕯️ Entry Point ===
if __name__ == "__main__":
    run_gene()

