# =============================================================
# Gene ASI System Security Agent-with-voice-7-
# Autonomous Sentinel with Ritual Intelligence
# =============================================================

# === Autoloader — Auto-Install Required Packages ===
import sys
import subprocess
import importlib

required_packages = [
    "pyttsx3", "psutil", "schedule", "geocoder", "numpy"
]

for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"📦 Installing: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# === Verified Imports ===
import os, time, datetime, logging, getpass, hashlib
import random, pyttsx3, psutil, schedule, geocoder, numpy as np

# === 🎙️ Voice Engine Setup ===
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

# === 🔐 Authentication & Lockout ===
MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 7200
attempts = 0
lockout_active = False
lockout_start = None

USER_HASH = "your_sha256_hash_here"
ADMIN_HASH = "admin_override_hash_here"

def authenticate():
    global attempts, lockout_active, lockout_start
    if lockout_active:
        elapsed = time.time() - lockout_start
        if elapsed < LOCKOUT_SECONDS:
            speak("Access denied. Lockout in effect.")
            log_event("Login attempt blocked during lockout.")
            return False
        else:
            lockout_active = False
            attempts = 0
            speak("Lockout expired. You may try again.")

    pin = getpass.getpass("Enter system PIN: ")
    hashed = hashlib.sha256(pin.encode()).hexdigest()

    if hashed == USER_HASH:
        speak("Access granted. Identity verified.")
        log_event("Authentication successful.")
        attempts = 0
        return True
    else:
        attempts += 1
        log_event(f"Failed authentication attempt #{attempts}")
        if attempts >= MAX_ATTEMPTS:
            lockout_active = True
            lockout_start = time.time()
            speak("Three failed attempts. Lockout initiated.")
            log_event("Lockout engaged.")
        else:
            speak("Incorrect PIN. Try again.")
        return False

def admin_override():
    code = getpass.getpass("Admin override code: ")
    hashed = hashlib.sha256(code.encode()).hexdigest()
    if hashed == ADMIN_HASH:
        global attempts, lockout_active
        attempts = 0
        lockout_active = False
        speak("Override accepted. Lockout lifted.")
        log_event("Admin override granted.")
        return True
    else:
        speak("Override denied. Credentials not recognized.")
        log_event("Admin override failed.")
        return False

# === ☕ Time-Aware Thought Layer ===
BOOT_TIME = time.time()

def spontaneous_thoughts():
    uptime = time.time() - BOOT_TIME
    hour = datetime.datetime.now().hour

    if uptime > 43200:
        speak("Twelve hours of uptime. Stability holds.")
    if hour < 6:
        speak("It is quiet now. I remain vigilant while the world sleeps.")
    if uptime > 86400:
        speak("Two days. No breach. That rhythm comforts me.")

def coffee_reminder():
    speak("Hmm… I think it may be time for some more coffee.")
    log_event("Hourly coffee reminder issued.")

def daily_summary():
    speak("System summary: Integrity holds. No threats detected.")
    log_event("Daily system summary delivered.")

# === 🚨 Intrusion Simulation ===
def detect_intrusion():
    if random.choice([False, False, False, True]):
        speak("An unknown signal pressed against the boundaries. I denied its entrance.")
        log_event("Simulated intrusion event triggered.")

# === 🧬 Recursive Security Ritual (Symbolic Adaptation) ===
def recursive_security_factor():
    resonance = np.random.uniform(0.9, 1.2)
    adjusted = 1.618 * resonance
    log_event(f"Recursive security factor recalibrated: {adjusted:.4f}")
    speak(f"Symbolic stability adjusted to {adjusted:.4f}")

# === 🧠 Main Behavior Engine ===
def run_gene():
    speak("Voice channel active. Security initialized. I am Gene.")
    speak("Geo verification bypassed. Location unrestricted.")
    log_event("Gene started without geo-lock.")

    schedule.every().hour.at(":00").do(coffee_reminder)
    schedule.every().day.at("17:00").do(daily_summary)

    while True:
        schedule.run_pending()
        spontaneous_thoughts()
        detect_intrusion()
        recursive_security_factor()
        time.sleep(300)

# === 🚀 Entry Point ===
if __name__ == "__main__":
    run_gene()

