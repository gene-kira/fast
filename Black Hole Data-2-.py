import tkinter as tk
import threading
import socket
import time
import json
from datetime import datetime, timedelta

# === OPTIONAL ENCRYPTION (AES) ===
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

def pad(s): return s + (16 - len(s) % 16) * chr(16 - len(s) % 16)
def unpad(s): return s[:-ord(s[len(s)-1:])]
def encrypt_payload(payload, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(payload).encode())
def decrypt_payload(enc_payload, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(enc_payload).decode())

# === CLIENT CLASS FOR INTEGRATION ===
class MagicBoxClient:
    def __init__(self, host='127.0.0.1', port=7070, encrypt=False, key=None):
        self.host = host
        self.port = port
        self.encrypt = encrypt and ENCRYPTION_AVAILABLE
        self.key = key or get_random_bytes(16) if self.encrypt else None

    def send(self, data, as_json=False):
        payload = json.dumps(data) if as_json else str(data)
        if self.encrypt:
            payload = encrypt_payload(payload, self.key)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.send(payload if self.encrypt else payload.encode())

# === GUI MAGICBOX APPLICATION ===
class MagicBoxGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MAGICBOX-OPS 🎩")
        self.root.geometry("750x500")

        self.log_clear_time = datetime.now() + timedelta(days=1)
        self.current_theme = "Cyberpunk"
        self.themes = {
            "Cyberpunk": {"bg": "#0f0f0f", "fg": "#ff00ff", "font": "Consolas", "button_bg": "#00ffff", "entry_bg": "#262626", "log_bg": "#000000"},
            "Noir": {"bg": "#2c2c2c", "fg": "#d3d3d3", "font": "Courier New", "button_bg": "#404040", "entry_bg": "#383838", "log_bg": "#111111"},
            "Retro Terminal": {"bg": "#000000", "fg": "#00ff00", "font": "Lucida Console", "button_bg": "#003300", "entry_bg": "#001100", "log_bg": "#000000"},
        }

        self.setup_ui()
        threading.Thread(target=self.auto_clear_log, daemon=True).start()
        threading.Thread(target=self.socket_listener, daemon=True).start()

    def setup_ui(self):
        t = self.themes[self.current_theme]
        self.root.configure(bg=t["bg"])

        # Theme menu
        menu = tk.Menu(self.root)
        theme_menu = tk.Menu(menu, tearoff=0)
        for theme in self.themes:
            theme_menu.add_command(label=theme, command=lambda tname=theme: self.switch_theme(tname))
        menu.add_cascade(label="Theme", menu=theme_menu)
        self.root.config(menu=menu)

        # Title
        self.title = tk.Label(self.root, text="MAGICBOX INTEL CONSOLE", font=(t["font"], 18), fg=t["fg"], bg=t["bg"])
        self.title.pack(pady=10)

        # Manual entry
        eframe = tk.Frame(self.root, bg=t["bg"])
        eframe.pack()
        self.entry_label = tk.Label(eframe, text="Alias:", font=(t["font"], 12), fg=t["fg"], bg=t["bg"])
        self.entry_label.pack(side=tk.LEFT)
        self.entry = tk.Entry(eframe, width=30, font=(t["font"], 12), bg=t["entry_bg"], fg=t["fg"])
        self.entry.pack(side=tk.LEFT, padx=5)
        self.send_btn = tk.Button(self.root, text="Send Manual", command=self.manual_send, font=(t["font"], 12), bg=t["button_bg"])
        self.send_btn.pack(pady=5)

        # Display + log
        self.data_frame = tk.Frame(self.root, bg=t["entry_bg"], bd=2, relief=tk.RIDGE)
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.log = tk.Text(self.root, height=6, bg=t["log_bg"], fg=t["fg"], font=(t["font"], 10))
        self.log.pack(fill=tk.BOTH, padx=20, pady=(0,10))
        self.log.insert(tk.END, "[🔒 LOG INITIATED]\n")

        # Status
        self.status = tk.Label(self.root, text="STATUS: Ready", font=(t["font"], 12), fg=t["fg"], bg=t["bg"])
        self.status.pack()

    def switch_theme(self, theme_name):
        self.current_theme = theme_name
        for w in self.root.winfo_children():
            w.destroy()
        self.setup_ui()

    def manual_send(self):
        name = self.entry.get().strip() or f"Manual-{int(time.time())}"
        self.display_payload(name)
        self.entry.delete(0, tk.END)

    def display_payload(self, content):
        t = self.themes[self.current_theme]
        label = tk.Label(self.data_frame, text=f"📡 {content}", font=(t["font"], 12), fg=t["fg"], bg=t["entry_bg"])
        label.pack(anchor='w', padx=10, pady=5)
        self.status.config(text=f"STATUS: Received → {content}")
        threading.Thread(target=self.self_destruct, args=(label, content), daemon=True).start()

    def self_destruct(self, label, name):
        time.sleep(30)
        label.destroy()
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{stamp}] {name} destroyed\n")
        self.log.see(tk.END)

    def auto_clear_log(self):
        while True:
            if datetime.now() > self.log_clear_time:
                self.log.delete('1.0', tk.END)
                self.log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Daily log cleared\n")
                self.log_clear_time = datetime.now() + timedelta(days=1)
            time.sleep(60)

    def socket_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 7070))
            s.listen()
            while True:
                conn, _ = s.accept()
                with conn:
                    raw = conn.recv(2048)
                    try:
                        if ENCRYPTION_AVAILABLE and raw.startswith(b'\x00'):
                            decrypted = decrypt_payload(raw[1:], client.key)
                            content = decrypted
                        else:
                            content = raw.decode()
                        try:
                            content = json.loads(content) if content.startswith('{') else content
                        except: pass
                        self.display_payload(str(content))
                    except Exception as e:
                        print("Error handling socket data:", e)

def main():
    root = tk.Tk()
    app = MagicBoxGUI(root)
    root.mainloop()

# Run GUI
if __name__ == "__main__":
    main()

