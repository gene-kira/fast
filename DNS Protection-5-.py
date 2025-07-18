import subprocess
import sys

# ✅ Auto-install required packages
def auto_install():
    libs = ['requests', 'Pillow']
    for lib in libs:
        try:
            __import__(lib)
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", lib])

    try:
        import tkinter
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", "tk"])
        print("✅ All required libraries installed. Please restart the program.")
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

# 📁 Memory File Path
MEMORY_FILE = "dns_guardian_config.json"
CLOUDFLARED_PATH = "cloudflared"

class DNSGuardian:
    def __init__(self, root):
        self.root = root
        self.root.title("DNS Guardian 🧓")
        self.monitoring = False
        self.config = self.load_memory()
        self.cloudflared_proc = None

        self.status_label = tk.Label(root, text="Status: Idle", fg="blue", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # 🖱️ One-click interface (friendly layout)
        actions = [
            ("Start Protection", self.start_protection),
            ("Stop Protection", self.stop_protection),
            ("Check for Leaks", self.check_leak),
            ("Toggle Encrypted DNS", self.toggle_encrypted_dns),
            ("Settings", self.open_settings)
        ]

        for label, action in actions:
            tk.Button(root, text=label, command=action, width=30, height=2, font=("Arial", 10)).pack(pady=3)

    # 🧠 Persistent Config
    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as file:
                return json.load(file)
        else:
            default = {
                "resolver": "1.1.1.1",
                "auto_monitor": True,
                "check_interval": 120,
                "encrypted_dns": False,
                "whitelist_regions": ["US", "CA", "UK"]
            }
            self.save_memory(default)
            return default

    def save_memory(self, config):
        with open(MEMORY_FILE, "w") as file:
            json.dump(config, file)

    # 🧪 Leak Check + Region Watch
    def check_leak(self):
        try:
            dns_ip = self.get_system_dns()
            public_ip = requests.get("https://api.ipify.org").text
            local_ip = socket.gethostbyname(socket.gethostname())

            leak = public_ip != local_ip
            foreign = self.is_foreign_dns(dns_ip)

            if leak or foreign:
                msg = f"Public IP: {public_ip}\nLocal IP: {local_ip}\nDNS Server: {dns_ip}"
                self.status_label.config(text="Status: DNS Leak or Foreign Detected", fg="red")
                messagebox.showwarning("DNS Issue", msg)
                self.fix_leak()
            else:
                self.status_label.config(text="Status: All Good ✅", fg="green")
                messagebox.showinfo("Safe", "No DNS issues detected.")
        except Exception as e:
            self.status_label.config(text="Status: Check Failed", fg="orange")
            messagebox.showerror("Error", str(e))

    def get_system_dns(self):
        try:
            output = subprocess.check_output("nslookup", shell=True).decode()
            for line in output.splitlines():
                if "Server:" in line:
                    return line.split("Server:")[1].strip()
        except Exception:
            return "Unknown"

    def is_foreign_dns(self, dns_ip):
        try:
            info = requests.get(f"http://ip-api.com/json/{dns_ip}").json()
            return info["countryCode"] not in self.config.get("whitelist_regions", [])
        except:
            return False

    # 🧹 DNS Healing
    def fix_leak(self):
        try:
            subprocess.call(["ipconfig", "/flushdns"])
            for iface in ["Ethernet", "Wi-Fi"]:
                subprocess.call(["netsh", "interface", "ip", "set", "dns", f"name={iface}", "static", self.config["resolver"]])
            messagebox.showinfo("Fix Applied", "DNS repaired to safe resolver.")
        except Exception as e:
            messagebox.showerror("Fix Failed", str(e))

    # 🔐 Tunnel Toggle
    def toggle_encrypted_dns(self):
        if self.config["encrypted_dns"]:
            self.stop_cloudflared()
            self.config["encrypted_dns"] = False
            self.status_label.config(text="Status: Unencrypted DNS", fg="orange")
        else:
            self.start_cloudflared()
        self.save_memory(self.config)

    def start_cloudflared(self):
        try:
            self.cloudflared_proc = subprocess.Popen(
                [CLOUDFLARED_PATH, "proxy-dns"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.config["encrypted_dns"] = True
            self.status_label.config(text="Status: Encrypted DNS Active 🔒", fg="blue")
        except Exception as e:
            self.config["encrypted_dns"] = False
            self.status_label.config(text="Status: Encrypted DNS Failed", fg="orange")
            messagebox.showwarning("Tunnel Error", str(e))
            self.fix_leak()

    def stop_cloudflared(self):
        if self.cloudflared_proc:
            self.cloudflared_proc.terminate()
            self.cloudflared_proc = None

    # 🧭 Background Guardian
    def start_protection(self):
        self.status_label.config(
            text="Status: Active Protection 🟢",
            fg="blue" if self.config["encrypted_dns"] else "orange"
        )
        self.monitoring = True
        if self.config["auto_monitor"]:
            threading.Thread(target=self.monitor_dns, daemon=True).start()

    def stop_protection(self):
        self.status_label.config(text="Status: Protection Stopped", fg="red")
        self.monitoring = False

    def monitor_dns(self):
        while self.monitoring:
            self.check_leak()
            time.sleep(self.config["check_interval"])

    # ⚙️ Settings Panel
    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings ⚙️")

        tk.Label(win, text="DNS Resolver:").pack(pady=5)
        resolver_entry = tk.Entry(win)
        resolver_entry.insert(0, self.config["resolver"])
        resolver_entry.pack()

        auto_monitor_var = tk.BooleanVar(value=self.config["auto_monitor"])
        tk.Checkbutton(win, text="Enable Auto Monitor", variable=auto_monitor_var).pack()

        tk.Label(win, text="Check Interval (sec):").pack(pady=5)
        interval_entry = tk.Entry(win)
        interval_entry.insert(0, str(self.config["check_interval"]))
        interval_entry.pack()

        tk.Label(win, text="Whitelisted Regions (comma-separated):").pack(pady=5)
        region_entry = tk.Entry(win)
        region_entry.insert(0, ",".join(self.config.get("whitelist_regions", [])))
        region_entry.pack()

        def save():
            try:
                self.config["resolver"] = resolver_entry.get()
                self.config["auto_monitor"] = auto_monitor_var.get()
                self.config["check_interval"] = int(interval_entry.get())
                self.config["whitelist_regions"] = [r.strip().upper() for r in region_entry.get().split(",")]
                self.save_memory(self.config)
                win.destroy()
            except:
                messagebox.showerror("Error", "Invalid entries.")

        tk.Button(win, text="Save Settings", command=save).pack(pady=10)

# 🧪 Launch
if __name__ == "__main__":
    root = tk.Tk()
    app = DNSGuardian(root)
    root.mainloop()

