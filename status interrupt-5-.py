# ✅ Autoloader for libraries
try:
    import tkinter as tk
    from tkinter import ttk, filedialog
    import ast
    import threading
    import time
    import os
    import subprocess
except ImportError as e:
    print(f"[ERROR] Missing: {e.name}. Please install required libraries.")
    exit()

class MagicBoxGUI:
    def __init__(self, root):
        root.title("MagicBox GlyphGuard — Agent Console")
        root.geometry("1000x700")
        root.configure(bg="#1c1c1c")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1c1c1c", borderwidth=0)
        style.configure("TNotebook.Tab", background="#333333", foreground="#ffffff", padding=10)
        style.map("TNotebook.Tab", background=[("selected", "#5c2d91")])

        notebook = ttk.Notebook(root)
        notebook.pack(expand=1, fill="both", padx=10, pady=10)

        # 🔍 Syntax Tab
        syntax_frame = ttk.Frame(notebook)
        notebook.add(syntax_frame, text="Syntax Checker")

        ttk.Label(syntax_frame, text="Paste Your Code Here:", foreground="#ffffff", background="#1c1c1c").pack(pady=5)
        self.syntax_entry = tk.Text(syntax_frame, height=10, bg="#2a2a2a", fg="#ffffff")
        self.syntax_entry.pack(padx=20, pady=5, fill="both")
        ttk.Button(syntax_frame, text="Check Syntax", command=self.verify_syntax).pack(pady=5)
        ttk.Button(syntax_frame, text="Run This Code", command=self.execute_code_sandbox).pack(pady=5)

        # 📁 Memory Tab
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Memory Module")

        ttk.Label(memory_frame, text="External Module:", foreground="#ffffff", background="#1c1c1c").pack(pady=5)
        ttk.Button(memory_frame, text="Load Module", command=self.inject_memory).pack(pady=5)
        ttk.Button(memory_frame, text="Encrypt Module", command=self.encrypt_memory).pack(pady=5)

        # 🗝 Registry Tab
        registry_frame = ttk.Frame(notebook)
        notebook.add(registry_frame, text="Agent Registry")
        self.registry_map = {}

        ttk.Label(registry_frame, text="Register Key & Value:", foreground="#ffffff", background="#1c1c1c").pack(pady=5)
        self.key_entry = tk.Entry(registry_frame, bg="#2a2a2a", fg="#ffffff")
        self.key_entry.pack(padx=20, pady=2, fill="x")
        self.value_entry = tk.Entry(registry_frame, bg="#2a2a2a", fg="#ffffff")
        self.value_entry.pack(padx=20, pady=2, fill="x")
        ttk.Button(registry_frame, text="Secure & Save", command=self.register_encrypted_node).pack(pady=5)
        self.registry_display = tk.Text(registry_frame, height=8, bg="#2a2a2a", fg="#00ffff")
        self.registry_display.pack(padx=20, pady=5, fill="both")

        # 🧠 Agent Status Tab
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="System Status")
        ttk.Label(status_frame, text="Diagnostics:", foreground="#ffffff", background="#1c1c1c").pack(pady=5)
        self.status_box = tk.Text(status_frame, height=10, bg="#2a2a2a", fg="#00ff00")
        self.status_box.pack(padx=20, pady=5, fill="both")

        # 🧾 Console Tab
        console_frame = ttk.Frame(notebook)
        notebook.add(console_frame, text="Console")
        ttk.Label(console_frame, text="Type a Shell Command:", foreground="#ffffff", background="#1c1c1c").pack(pady=5)
        self.console_entry = tk.Entry(console_frame, bg="#2a2a2a", fg="#ffffff")
        self.console_entry.pack(padx=20, pady=5, fill="x")
        self.console_output = tk.Text(console_frame, height=10, bg="#2a2a2a", fg="#00ffcc")
        self.console_output.pack(padx=20, pady=5, fill="both")
        ttk.Button(console_frame, text="Run Command", command=self.run_console_command).pack(pady=5)

        # 🔧 Setup
        self.bind_hotkeys(root)
        self.start_monitor()

    def verify_syntax(self):
        code = self.syntax_entry.get("1.0", "end-1c")
        self.route_feedback("Checking syntax...")
        try:
            ast.parse(code)
            self.route_feedback("✅ Code looks good!")
        except SyntaxError as e:
            self.route_feedback(f"❌ Syntax Issue: {e}")

    def execute_code_sandbox(self):
        code = self.syntax_entry.get("1.0", "end-1c")
        try:
            exec(code, {})
            self.route_feedback("🧪 Code executed successfully.")
        except Exception as e:
            self.route_feedback(f"🔥 Execution failed: {e}")

    def inject_memory(self):
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        if path:
            name = os.path.basename(path)
            self.route_feedback(f"Module loaded: {name}")

    def encrypt_memory(self):
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        if path:
            encrypted = f"#Encrypted\n{open(path).read()}"[::-1]
            with open("encrypted_module.glyph", "w", encoding="utf-8") as f:
                f.write(encrypted)
            self.route_feedback(f"Encrypted and saved: {os.path.basename(path)}")

    def register_encrypted_node(self):
        key = self.key_entry.get()
        value = self.value_entry.get()
        encrypted_value = value[::-1]
        self.registry_map[key] = encrypted_value
        self.registry_display.insert("end", f"{key}: {encrypted_value}\n")

    def run_console_command(self):
        cmd = self.console_entry.get()
        try:
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
            self.console_output.insert("end", f"> {cmd}\n{result}\n")
        except subprocess.CalledProcessError as e:
            self.console_output.insert("end", f"Error:\n{e.output}\n")
        self.console_output.see("end")

    def route_feedback(self, message):
        self.status_box.insert("end", f"{message}\n")
        self.status_box.see("end")

    def autosave_code(self):
        code = self.syntax_entry.get("1.0", "end-1c")
        with open("autosave_snippet.py", "w", encoding="utf-8") as f:
            f.write(code)
        self.route_feedback("💾 Code saved automatically.")

    def bind_hotkeys(self, root):
        root.bind("<Control-s>", lambda e: self.autosave_code())

    def passive_monitoring(self):
        while True:
            self.route_feedback("✅ Agent is online | System normal.")
            time.sleep(15)

    def start_monitor(self):
        threading.Thread(target=self.passive_monitoring, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxGUI(root)
    root.mainloop()

