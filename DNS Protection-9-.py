import subprocess
import sys

# ✅ Auto-install required packages
def auto_install():
    libs = ['requests', 'Pillow', 'pyttsx3']
    for lib in libs:
        try:
            __import__(lib)
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", lib])
    try:
        import tkinter
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", "tk"])
        print("✅ All libraries installed. Please restart.")
        sys.exit()

auto_install()

# 🔁 Re-import safely
import tkinter as tk
from tkinter import messagebox
import requests
import json
import os
import threading
import time
import socket
from PIL import Image, ImageDraw
import pyttsx3

MEMORY_FILE = "dns_guardian_config.json"
CLOUDFLARED_PATH = "cloudflared"

class DNSGuardian:
    def __init__(self, root):
        self.root = root
        self.root.title("DNS Guardian 🧓")
        self.monitoring = False
        self.config = self.load_memory()
        self.cloudflared_proc = None
        self.voice_engine = pyttsx3.init()

        self.status_label = tk.Label(root, text="Status: Idle", fg="blue", font=("Arial", 12))
        self.status_label.pack(pady=10)

        for text, command in [
            ("Start Protection", self.start_protection),
            ("Stop Protection", self.stop_protection),
            ("Check for Leaks", self.check_leak),
            ("Toggle Encrypted DNS", self.toggle_encrypted_dns),
            ("Settings", self.open_settings)
        ]:
            tk.Button(root, text=text, command=command, width=30, height=2, font=("Arial", 10)).pack(pady=3)

    def speak(self, message):
        try:
            self.voice_engine.say(message)
            self.voice_engine.runAndWait()
        except:
            pass

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        config = {
            "resolver": "1.1.1.1",
            "auto_monitor": True,
            "check_interval": 120,
            "ping_interval": 10,
            "encrypted_dns": False,
            "whitelist_regions": ["US", "CA", "UK"]
        }
        self.save_memory(config)
        return config

    def save_memory(self, config):
        with open(MEMORY_FILE, "w") as f:
            json.dump(config, f)

    def get_system_dns(self):
        try:
            out = subprocess.check_output("nslookup", shell=True, timeout=5).decode()
            for line in out.splitlines():
                if "Server:" in line:
                    return line.split("Server:")[1].strip()
        except:
            return "Unknown"

    def is_foreign_dns(self, dns_ip):
        try:
            info = requests.get(f"http://ip-api.com/json/{dns_ip}", timeout=5).json()
            return info.get("countryCode") not in self.config.get("whitelist_regions", [])
        except:
            return False

    def fast_dns_check(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            result = s.connect_ex(("1.1.1.1", 53))
            s.close()
            return result == 0
        except:
            return False

    def check_leak(self):
        try:
            dns_ip = self.get_system_dns()
            public_ip = requests.get("https://api.ipify.org", timeout=5).text
            local_ip = socket.gethostbyname(socket.gethostname())
            leak = public_ip != local_ip
            foreign = self.is_foreign_dns(dns_ip)

            if leak or foreign:
                self.status_label.config(text="Leak or Foreign DNS Detected", fg="red")
                self.speak("DNS leak or foreign server detected.")
                messagebox.showwarning("DNS Issue", f"Public IP: {public_ip}\nLocal IP: {local_ip}\nDNS: {dns_ip}")
                self.fix_leak()
            else:
                self.status_label.config(text="Status: All Good ✅", fg="green")
        except requests.exceptions.Timeout:
            self.status_label.config(text="Leak Test Timeout ⏱️", fg="orange")
            self.speak("Leak test timed out.")
        except Exception as e:
            self.status_label.config(text="Leak Test Failed", fg="orange")
            self.speak("Leak test error.")

    def fix_leak(self):
        try:
            subprocess.call(["ipconfig", "/flushdns"])
            for iface in ["Ethernet", "Wi-Fi"]:
                subprocess.call(["netsh", "interface", "ip", "set", "dns", f"name={iface}", "static", self.config["resolver"]])
            self.speak("DNS settings restored.")
        except:
            self.speak("Failed to repair DNS.")

    def start_cloudflared(self):
        try:
            self.cloudflared_proc = subprocess.Popen(
                [CLOUDFLARED_PATH, "proxy-dns"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e:
            messagebox.showwarning("Tunnel Failed", str(e))
            return False

    def stop_cloudflared(self):
        if self.cloudflared_proc:
            self.cloudflared_proc.terminate()
            self.cloudflared_proc = None

    def toggle_encrypted_dns(self):
        self.stop_cloudflared()
        self.config["encrypted_dns"] = False

        if self.start_cloudflared():
            self.config["encrypted_dns"] = True
            self.status_label.config(text="Encrypted DNS Active 🔒", fg="blue")
            self.speak("Encrypted DNS tunnel started.")
        else:
            self.status_label.config(text="Fallback to System DNS 🔄", fg="orange")
            self.speak("Encrypted DNS failed. Using fallback.")
            self.fix_leak()

        self.save_memory(self.config)

    def start_protection(self):
        self.status_label.config(text="Protection Active 🟢", fg="blue" if self.config["encrypted_dns"] else "orange")
        self.monitoring = True
        self.speak("DNS protection started.")
        self.pulse_label()
        threading.Thread(target=self.watchdog_loop, daemon=True).start()

    def stop_protection(self):
        self.status_label.config(text="Protection Stopped", fg="red")
        self.monitoring = False
        self.speak("DNS protection stopped.")

    def watchdog_loop(self):
        while self.monitoring:
            if not self.fast_dns_check():
                self.status_label.config(text="DNS Heartbeat Lost 💔", fg="red")
                self.speak("DNS heartbeat lost. Checking now.")
                self.check_leak()
            else:
                self.status_label.config(text="Monitoring OK 💙", fg="blue")
            time.sleep(self.config.get("ping_interval", 10))

    def pulse_label(self):
        def pulse():
            colors = ["#4488FF", "#2255CC"]
            while self.monitoring:
                for c in colors:
                    self.status_label.config(fg=c)
                    time.sleep(0.6)
        threading.Thread(target=pulse, daemon=True).start()

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings ⚙️")

        tk.Label(win, text="DNS Resolver:").pack()
        resolver_entry = tk.Entry(win)
        resolver_entry.insert(0, self.config["resolver"])
        resolver_entry.pack()

        auto_monitor_var = tk.BooleanVar(value=self.config["auto_monitor"])
        tk.Checkbutton(win, text="Enable Auto Monitor", variable=auto_monitor_var).pack()

        tk.Label(win, text="Check Interval (sec):").pack()
        interval_entry = tk.Entry(win)
        interval_entry.insert(0, str(self.config["check_interval"]))
        interval_entry.pack()

        tk.Label(win, text="Ping Interval (sec):").pack()
        ping_entry = tk.Entry(win)
        ping_entry.insert(0, str(self.config.get("ping_interval", 10)))
        ping_entry.pack()

        tk.Label(win, text="Whitelisted Regions:").pack()
        region_entry = tk.Entry(win)
        region_entry.insert(0, ",".join(self.config.get("whitelist_regions", [])))
        region_entry.pack()

        def save():
            try:
                self.config["resolver"] = resolver_entry.get()
                self.config["auto_monitor"] = auto_monitor_var.get()
                self.config["check_interval"] = int(interval_entry.get())
                self.config["ping_interval"] = int(ping_entry.get())
                self.config["whitelist_regions"] = [r.strip().upper() for r in region_entry.get().split(",")]
                self.save_memory(self.config)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid settings.\n{e}")
                self.speak("Settings could not be saved.")

        tk.Button(win, text="Save Settings", command=save).pack(pady=10)

# 🧪 Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DNSGuardian(root)
    root.mainloop()

