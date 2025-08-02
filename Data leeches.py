import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# 🔧 Auto-install required packages if missing
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# 🎩 MagicBox GUI
class MagicBoxScanner:
    def __init__(self, root):
        root.title("🧙 MagicBox - Data Sniffer")
        root.geometry("600x400")
        root.configure(bg="#2e2e3a")
        
        self.status_label = tk.Label(root, text="Ready to scan for sneaky apps!", fg="white", bg="#2e2e3a", font=("Helvetica", 14))
        self.status_label.pack(pady=20)

        self.scan_button = ttk.Button(root, text="🕵️‍♂️ One-Click Scan", command=self.run_scan)
        self.scan_button.pack(pady=10)

        self.result_box = tk.Text(root, bg="#1e1e2f", fg="#00ffcc", font=("Courier", 10), height=15, wrap="word")
        self.result_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), foreground='#2e2e3a', background='#00ffcc')

    def run_scan(self):
        self.status_label.config(text="Scanning...")
        self.result_box.delete("1.0", tk.END)
        thread = threading.Thread(target=self.scan_processes)
        thread.start()

    def scan_processes(self):
        suspicious_keywords = ['steal', 'keylog', 'exfil', 'snoop', 'spy', 'scraper', 'grabber']
        results = []

        for proc in psutil.process_iter(['name', 'exe']):
            try:
                pname = proc.info['name'] or ''
                ppath = proc.info['exe'] or ''
                match = any(kw in pname.lower() or kw in ppath.lower() for kw in suspicious_keywords)
                if match:
                    results.append(f"🔴 {pname}\n    Path: {ppath}\n")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if results:
            self.result_box.insert(tk.END, "\n".join(results))
            self.status_label.config(text=f"Found {len(results)} suspect processes!")
        else:
            self.result_box.insert(tk.END, "✅ No known suspicious processes detected.")
            self.status_label.config(text="All clear!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxScanner(root)
    root.mainloop()

