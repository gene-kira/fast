import subprocess
import sys
import os
import time
import threading
import tkinter as tk

# === 🧰 Auto-Install Required Libraries ===
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required = {
    "pyttsx3": "pyttsx3",
    "PIL": "Pillow"
}

for module, pip_name in required.items():
    try:
        __import__(module)
    except ImportError:
        print(f"Installing {pip_name}...")
        install(pip_name)

# === Imports after install ===
import pyttsx3
from PIL import Image

# === 🔊 Startup Voice ===
def startup_voice():
    engine = pyttsx3.init('sapi5')  # Use 'espeak' on Linux
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say("Data self-destruct mechanism engaged.")
    engine.runAndWait()

# === 🧪 Simulated Image Validation ===
def validate_image(image_path):
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        pixels = list(img.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        return avg_brightness > 100  # threshold for "valid"
    except Exception:
        return False

# === 🖼️ GUI Class ===
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "sentinel_log.txt")

class MagicBoxSentinel:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicBox Sentinel — Agent Ember")
        self.root.configure(bg="#121212")
        self.root.geometry("600x400")

        os.makedirs(LOG_DIR, exist_ok=True)
        self.auto_cleanup_logs()

        self.status_label = tk.Label(root, text="Monitoring active...", font=("Courier", 16),
                                     bg="#121212", fg="cyan")
        self.status_label.pack(pady=10)

        self.log_box = tk.Text(root, height=12, bg="#1e1e2f", fg="lime", insertbackground='white')
        self.log_box.pack(fill="both", padx=10, pady=5)

        self.clear_button = tk.Button(root, text="🧹 Clear Logs", command=self.clear_logs,
                                      bg="black", fg="white")
        self.clear_button.pack(pady=5)

        self.start_monitoring()

    def start_monitoring(self):
        threading.Thread(target=self.monitor_loop, daemon=True).start()

    def monitor_loop(self):
        while True:
            image_path = "sample.jpg"
            result = validate_image(image_path)

            if result:
                self.log("✅ Valid image detected (brightness OK)")
                self.voice_feedback("Validation complete. Integrity confirmed.")
            else:
                self.log("⚠️ Suspicious image detected (brightness low)")
                self.voice_feedback("Warning: Data anomaly detected. Initiating purge.")

            time.sleep(30)
            self.log("🔥 Data invalidated after timeout.")

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        self.log_box.insert(tk.END, full_message + "\n")
        self.log_box.see(tk.END)

        with open(LOG_FILE, "a") as f:
            f.write(full_message + "\n")

    def voice_feedback(self, message):
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 1.0)
        engine.setProperty('voice', engine.getProperty('voices')[0].id)
        engine.say(message)
        engine.runAndWait()

    def clear_logs(self):
        if os.path.exists(LOG_FILE):
            open(LOG_FILE, 'w').close()
        self.log("🧹 Logs cleared.")
        self.voice_feedback("Log vault purged.")

    def auto_cleanup_logs(self):
        if os.path.exists(LOG_FILE):
            modified = os.path.getmtime(LOG_FILE)
            if time.time() - modified > 86400:
                os.remove(LOG_FILE)
                self.log("🧨 Old logs auto-deleted.")
                self.voice_feedback("Old records eliminated.")

# === 🚀 Launch Everything ===
if __name__ == "__main__":
    startup_voice()
    root = tk.Tk()
    app = MagicBoxSentinel(root)
    root.mainloop()

