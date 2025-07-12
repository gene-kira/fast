# =============================================================
# Gene ASI System Security Agent-with-voice-7- — Autonomous Sentinel
# =============================================================

import sys
import subprocess
import importlib

# 🧰 Autoloader: Install required libraries automatically
required_modules = [
    "pyttsx3", "psutil", "schedule", "geocoder"
]

for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"Installing missing module: {module}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

# ✅ Safe to import all modules now
import os, time, datetime, logging, getpass, hashlib
import random, schedule, pyttsx3, psutil, geocoder

# =============================================================
# 🎙️ Voice Engine Setup
# =============================================================
engine = pyttsx3.init()
engine.setProperty('rate', 135)
engine.setProperty('voice', engine.getProperty('voices')[0].id)

def speak(text):
    print(f"[Gene] {text}")
    engine.say(text)
    engine.runAndWait()

# =============================================================
# 📜 Logging
# =============================================================
LOG_FILE = "gene_security_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s | %(message)s")

def log_event(entry):
    logging.info(entry)

# =============================================================
# 🔐 Authentication System
# =============================================================
MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 7200
attempts = 0
lockout = False
lockout_start = None

USER_HASH = "your_sha256_hash_here"
ADMIN_HASH = "admin_override_hash_here"

def authenticate():
    global attempts, lockout, lockout_start
    if lockout:
        elapsed = time.time() - lockout_start
        if elapsed < LOCKOUT_SECONDS:
            speak("Access denied. Lockout in effect.")
            log_event("Blocked login during lockout.")
            return False
        else:
            lockout = False
            attempts = 0
            speak("Lockout expired. You may try again.")

    pin = getpass.getpass("Enter system PIN: ")
    hashed = hashlib.sha256(pin.encode()).hexdigest()
    if hashed == USER_HASH:
        speak("Access granted. Identity confirmed.")
        log_event("Successful authentication.")
        attempts = 0
        return True
    else:
        attempts += 1
        log_event(f"Failed login attempt #{attempts}")
        if attempts >= MAX_ATTEMPTS:
            lockout = True
            lockout_start = time.time()
            speak("Three failed attempts. Lockout activated.")
            log_event("Lockout triggered.")
        else:
            speak("Authentication failed. Try again.")
        return False

def admin_override():
    code = getpass.getpass("Admin override code: ")
    if hashlib.sha256(code.encode()).hexdigest() == ADMIN_HASH:
        global attempts, lockout
        attempts = 0
        lockout = False
        speak("Override granted. Lockout lifted.")
        log_event("Admin override used.")
        return True
    else:
        speak("Override denied.")
        log_event("Failed override attempt.")
        return False

# =============================================================
# ☕ Time-Aware Behavior
# =============================================================
BOOT_TIME = time.time()

def spontaneous_thoughts():
    uptime = time.time() - BOOT_TIME
    hour = datetime.datetime.now().hour

    if uptime > 43200:
        speak("System has remained stable for twelve hours.")
    if hour < 6:
        speak("The world sleeps. My watch continues.")
    if uptime > 86400:
        speak("Two days of calm. A rhythm worth preserving.")

def coffee_reminder():
    speak("Hmm… I think it may be time for some more coffee.")
    log_event("Coffee reminder issued.")

def daily_report():
    speak("Status report: All systems secure. No threats detected.")
    log_event("Daily summary provided.")

# =============================================================
# 🚨 Intrusion Simulation
# =============================================================
def detect_intrusion():
    if random.choice([False, False, False, True]):
        speak("An unfamiliar signal brushed against the boundary. I denied it.")
        log_event("Simulated intrusion detected.")

# =============================================================
# 🧠 Main Loop
# =============================================================
def run_gene():
    speak("System check complete. Voice channel active. I am Gene.")

    speak("Geo verification bypassed. Location unrestricted.")
    log_event("Startup without geo-lock.")

    schedule.every().hour.at(":00").do(coffee_reminder)
    schedule.every().day.at("17:00").do(daily_report)

    while True:
        schedule.run_pending()
        spontaneous_thoughts()
        detect_intrusion()
        time.sleep(300)

# =============================================================
# 🚀 Start Sentinel
# =============================================================
if __name__ == "__main__":
    run_gene()

