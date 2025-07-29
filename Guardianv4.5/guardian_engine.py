import os
import time
from cryptography.fernet import Fernet

class MagicBoxGuardian:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.active_persona = "Baurdan"
        self.badges = []
        self.logs = []
        self.encrypted_files = []
        self.override_keys = {"Baurdan": "XK-99"}
        self.threat_ips = {}
        print("[SYSTEM] Guardian initialized. Persona locked to Baurdan.")

    def encrypt_file(self, filepath):
        with open(filepath, 'rb') as file:
            data = file.read()
        encrypted_data = self.cipher.encrypt(data)
        enc_path = filepath + ".enc"
        with open(enc_path, 'wb') as file:
            file.write(encrypted_data)
        self.encrypted_files.append(enc_path)
        self._log_event(f"Encrypted: {filepath}")
    
    def decrypt_file(self, enc_path):
        with open(enc_path, 'rb') as file:
            encrypted_data = file.read()
        data = self.cipher.decrypt(encrypted_data)
        dec_path = enc_path.replace(".enc", "_decrypted")
        with open(dec_path, 'wb') as file:
            file.write(data)
        self._log_event(f"Decrypted: {enc_path}")
    
    def _log_event(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.logs.append(f"[{timestamp}] {message}")
        print(f"[LOG] {message}")

    def detect_threat(self, ip):
        if ip not in self.threat_ips:
            self.threat_ips[ip] = {"count": 1}
            self._log_event(f"New IP detected: {ip}")
        else:
            self.threat_ips[ip]["count"] += 1
            if self.threat_ips[ip]["count"] > 5:
                self._log_event(f"Threat escalation from IP: {ip}")

    def unlock_badge(self, badge_name):
        if badge_name not in self.badges:
            self.badges.append(badge_name)
            self._log_event(f"Badge Unlocked: {badge_name}")
            print(f"[BADGE] {badge_name} now active.")

    def persona_override(self, persona, key_attempt):
        if persona in self.override_keys and self.override_keys[persona] == key_attempt:
            self.active_persona = persona
            self._log_event(f"Persona override: {persona}")
        else:
            self._log_event(f"Override failed for {persona}")

    def export_logs(self):
        with open("guardian_logs.txt", 'w') as file:
            file.write("\n".join(self.logs))
        print("[SYSTEM] Logs exported to guardian_logs.txt")

    def show_status(self):
        return {
            "active_persona": self.active_persona,
            "badges": self.badges,
            "threat_ips": self.threat_ips,
            "encrypted_files": self.encrypted_files,
            "log_count": len(self.logs)
        }

