# === Arkforge Guardian v7.1.5-dev — Trust + Voice Control Upgrade ===

# AUTO-INSTALL LIBRARIES
import sys, subprocess
required = ["uuid", "time", "random", "hashlib", "threading", "pyttsx3",
            "flask", "tkinter", "cryptography", "socket", "psutil"]
for lib in required:
    try: __import__(lib)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# IMPORTS
import uuid, time, random, threading, socket, psutil
import tkinter as tk
from tkinter import ttk, scrolledtext
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet
import pyttsx3

# UPGRADED VOICE SYSTEM
def speak(msg):
    try:
        # Check vault flag before speaking
        if not vault.trust_data.get("voice_alerts", True):
            return
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    except:
        print(f"[voice] {msg}")

# ENCRYPTED VAULT W/ TRUSTED IPS AND VOICE CONTROL
class Vault:
    def __init__(self, key_file="vault.key", data_file="trust.db"):
        self.key_file, self.data_file = key_file, data_file
        self.key = self.load_key()
        self.fernet = Fernet(self.key)
        self.trust_data = self.load_data()

        # Default trusted loopback IPs
        self.trust_data.setdefault("trusted_ips", {"127.0.0.1", "::1", "localhost"})
        # Default voice alerts ON
        self.trust_data.setdefault("voice_alerts", True)

    def load_key(self):
        try:
            with open(self.key_file, "rb") as f: return f.read()
        except:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f: f.write(key)
            return key

    def load_data(self):
        try:
            with open(self.data_file, "rb") as f:
                return eval(self.fernet.decrypt(f.read()).decode())
        except:
            return {}

    def save(self):
        encrypted = self.fernet.encrypt(str(self.trust_data).encode())
        with open(self.data_file, "wb") as f: f.write(encrypted)

# CORE KERNEL
class ASIKernel:
    def __init__(self, vault):
        self.id = str(uuid.uuid4())
        self.vault = vault
        self.memory, self.filters, self.swarm = {}, [], []
        self.glyphs, self.trust = {}, vault.trust_data

    def register_glyph(self, sig, fn): self.glyphs[sig] = fn
    def remember(self, k, v): self.memory[k] = v
    def recall(self, k): return self.memory.get(k)
    def install_filter(self, fn): self.filters.append(fn)
    def evaluate(self, signal): return all(f(signal) for f in self.filters)

# REFLEX LAYER WITH VOICE + TRUSTED IPS
class ReflexLayer:
    def __init__(self, kernel): self.kernel = kernel

    def evaluate(self, packet):
        port = str(packet.get("port", "null"))
        ip = packet.get("ip", "unknown")
        entropy = float(packet.get("entropy", 0))
        quarantine = self.kernel.recall("quarantine") or set()

        trusted_ips = self.kernel.vault.trust_data.get("trusted_ips", set())
        if ip in trusted_ips:
            return True  # Skip alert if trusted

        if port in quarantine:
            return True

        if entropy > 0.9:
            quarantine.add(port)
            self.kernel.remember("quarantine", quarantine)
            self.kernel.remember("last_threat_ip", ip)
            print(f"⚠️ Intruder detected on port {port} from IP {ip}")
            speak(f"Intruder on port {port} from {ip}")
            return self.operator_prompt(port)
        return True

    def operator_prompt(self, port):
        trust = self.kernel.trust.get(port, 1.0)
        ip = self.kernel.recall("last_threat_ip") or "unknown"

        for _ in range(3):
            print(f"\n🚨 Threat on port {port} (IP: {ip}) — Options:")
            print("[1] False Alarm  [2] Confirm Threat  [3] Terminate")
            choice = input(">> ").strip()
            if choice == "1":
                speak("Dismissed.")
                trust = min(1.0, trust + 0.05)
                self.ask_trust_ip(ip)
                break
            elif choice == "2":
                speak("Confirmed threat.")
                self.kernel.glyphs.get("Σ", lambda: None)()
                trust = max(0.4, trust - 0.2)
                break
            elif choice == "3":
                speak("Intruder eliminated.")
                self.kernel.glyphs.get("Σ", lambda: None)()
                trust = max(0.2, trust - 0.3)
                break
            else:
                print("Invalid.")
        else:
            speak("Auto-neutralizing.")
            self.kernel.glyphs.get("Σ", lambda: None)()
            trust = max(0.1, trust - 0.4)

        self.kernel.trust[port] = trust
        self.kernel.vault.trust_data = self.kernel.trust
        self.kernel.vault.save()
        return True

    def ask_trust_ip(self, ip):
        answer = input(f"🧠 Trust IP {ip} permanently? [y/N] ").strip().lower()
        if answer == "y":
            self.kernel.vault.trust_data["trusted_ips"].add(ip)
            self.kernel.vault.save()
            print(f"✅ {ip} added to trusted IPs.")

# SIGIL ORCHESTRATOR (now includes trust_ip & untrust_ip)
class SigilOrchestrator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.routes = {
            "Ω": self.defend, "omega": self.defend,
            "Ψ": self.resonate, "psi": self.resonate,
            "Σ": self.terminate, "sigma": self.terminate,
            "Φ": self.purge_ips,
            "trust_ip": self.trust_ip,
            "untrust_ip": self.untrust_ip,
            "voice_toggle": self.toggle_voice
        }

    def route(self, sigil): return self.routes.get(sigil.lower(), self.unknown)()

    def defend(self): print("🛡 Defense activated."); return True
    def resonate(self): print("🔊 Resonance initiated."); return True
    def terminate(self): print("⚖ Termination ritual cast."); speak("Threat terminated"); return True
    def purge_ips(self): IPMonitor(self.kernel).purge_unidentified(); return True

    def trust_ip(self, ip=None):
        ip = ip or self.kernel.recall("last_threat_ip")
        if ip:
            self.kernel.vault.trust_data["trusted_ips"].add(ip)
            self.kernel.vault.save()
            print(f"✅ {ip} trusted.")
        return True

    def untrust_ip(self, ip=None):
        ip = ip or self.kernel.recall("last_threat_ip")
        if ip and ip in self.kernel.vault.trust_data["trusted_ips"]:
            self.kernel.vault.trust_data["trusted_ips"].remove(ip)
            self.kernel.vault.save()
            print(f"❌ {ip} untrusted.")
        return True

    def toggle_voice(self):
        current = self.kernel.vault.trust_data.get("voice_alerts", True)
        self.kernel.vault.trust_data["voice_alerts"] = not current
        self.kernel.vault.save()
        state = "ON" if not current else "OFF"
        print(f"🔈 Voice alerts toggled {state}")
        return True

    def unknown(self): print("❓ Unknown sigil"); return False

# FLASK API WITH SIGIL PARAMS
flask_app = Flask(__name__)
sigil_queue = []

@flask_app.route("/sigil/<symbol>", methods=["POST"])
def receive_sigil(symbol):
    value = request.args.get("value")
    sigil_queue.append((symbol, value))
    return jsonify({"status": "received", "sigil": symbol, "value": value})

# SWARM + NETWORK SIMULATION
class SwarmAgent:
    def __init__(self, ident, reflex_fn): self.id, self.reflex = ident, reflex_fn
    def observe(self, packet): self.reflex(self.id, packet)

def deploy_swarm(kernel, count=3):
    def reflex(agent_id, packet): kernel.evaluate(packet)
    kernel.swarm = [SwarmAgent(f"sentinel_{i}", reflex) for i in range(count)]

def network_sim(kernel):
    while True:
        pkt = {
            "entropy": random.uniform(0.85, 1.1),
            "port": str(random.choice([22, 443, 8080, 3306, 31337])),
            "ip": f"192.168.1.{random.randint(2, 254)}"
        }
        kernel.evaluate(pkt)
        time.sleep(random.uniform(5, 9))

# HUD INTERFACE WITH VOICE TOGGLE
class ArkforgeHUD:
    def __init__(self, kernel):
        self.kernel = kernel
        self.root = tk.Tk()
        self.root.title("Arkforge Guardian v7.1.5-dev")
        self.root.geometry("680x460")
        self.build_gui()

    def build_gui(self):
        ttk.Label(self.root, text="Arkforge HUD", font=("Arial", 16)).pack(pady=10)
        self.logbox = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=75, height=18)
        self.logbox.pack(padx=10, pady=5)

        btn = ttk.Button(self.root, text="Toggle Voice Alerts", command=self.toggle_voice)
        btn.pack(pady=5)

        self.update_loop()

    def toggle_voice(self):
        orchestrator = SigilOrchestrator(self.kernel)
        orchestrator.toggle_voice()

    def update_loop(self):
        trust = self.kernel.trust
        ips = self.kernel.vault.trust_data.get("trusted_ips", set())
        voice = self.kernel.vault.trust_data.get("voice_alerts", True)

        self.logbox.insert(tk.END, f"🧠 Voice Alerts: {'ON' if voice else 'OFF'}\n")
        self.logbox.insert(tk.END, f"🌐 Trust Table: {trust}\n")
        self.logbox.insert(tk.END, f"🔓 Trusted IPs: {', '.join(sorted(ips))}\n\n")
        self.logbox.see(tk.END)
        self.root.after(8000, self.update_loop)

    def launch(self):
        self.root.mainloop()

# === FINAL BOOTLOADER ===
if __name__ == "__main__":
    print("🔧 Booting Arkforge Guardian v7.1.5-dev")

    # Initialize Vault & Core Kernel
    vault = Vault()
    kernel = ASIKernel(vault)

    # Install Reflex Filter
    reflex = ReflexLayer(kernel)
    kernel.install_filter(reflex.evaluate)

    # Register Sigil Rituals
    orchestrator = SigilOrchestrator(kernel)
    for sigil in orchestrator.routes:
        kernel.register_glyph(sigil, orchestrator.routes[sigil])

    # Sigil Queue Processor (for remote /sigil/x POSTs)
    def sigil_daemon():
        while True:
            if sigil_queue:
                symbol, value = sigil_queue.pop(0)
                fn = orchestrator.routes.get(symbol.lower())
                if fn:
                    if value:
                        fn(value)
                    else:
                        fn()
            time.sleep(1)
    threading.Thread(target=sigil_daemon, daemon=True).start()

    # Deploy Swarm and Network Simulation
    deploy_swarm(kernel)
    threading.Thread(target=network_sim, args=(kernel,), daemon=True).start()

    # Launch Flask API for remote sigils
    threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=5050), daemon=True).start()

    # Launch Guardian HUD
    ArkforgeHUD(kernel).launch()

