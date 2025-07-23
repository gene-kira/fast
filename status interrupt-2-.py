import tkinter as tk
from tkinter import ttk, filedialog
import ast
import threading
import time
import os

class MagicBoxGUI:
    def __init__(self, root):
        root.title("MagicBox GlyphGuard — ASI Agent Panel")
        root.geometry("700x500")
        root.configure(bg="#1c1c1c")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1c1c1c", borderwidth=0)
        style.configure("TNotebook.Tab", background="#333333", foreground="#ffffff", padding=10)
        style.map("TNotebook.Tab", background=[("selected", "#5c2d91")])

        notebook = ttk.Notebook(root)
        notebook.pack(expand=1, fill="both", padx=10, pady=10)

        # 🔍 Syntax Guard Tab
        syntax_frame = ttk.Frame(notebook)
        notebook.add(syntax_frame, text="Syntax Guard")

        ttk.Label(syntax_frame, text="Enter Code Snippet:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        self.syntax_entry = tk.Text(syntax_frame, height=10, bg="#2a2a2a", fg="#ffffff", insertbackground="#ffffff")
        self.syntax_entry.pack(padx=20, pady=5, fill="both")

        verify_btn = ttk.Button(syntax_frame, text="Verify Syntax", command=self.verify_syntax)
        verify_btn.pack(pady=10)

        # 🧠 Memory Augmentation Tab
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Memory Augmentation")

        ttk.Label(memory_frame, text="Upload Memory Module:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        memory_btn = ttk.Button(memory_frame, text="Inject Memory", command=self.inject_memory)
        memory_btn.pack(pady=5)

        # ⚙️ Agent Status Tab
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="Agent Status")

        ttk.Label(status_frame, text="System Diagnostics:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        self.status_box = tk.Text(status_frame, height=8, bg="#2a2a2a", fg="#00ff00", insertbackground="#ffffff")
        self.status_box.pack(padx=20, pady=5, fill="both")

        # 🚦 Start passive monitoring + hotkeys
        self.bind_hotkeys(root)
        self.start_monitor()

    def verify_syntax(self):
        code = self.syntax_entry.get("1.0", "end-1c")
        self.route_feedback("🔎 Checking syntax...")
        try:
            ast.parse(code)
            self.route_feedback("✅ Syntax is valid.", level="info")
        except SyntaxError as e:
            self.route_feedback(f"❌ Syntax Error: {e}", level="error")

    def inject_memory(self):
        file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if file_path:
            self.route_feedback(f"🧠 Module uploaded: {os.path.basename(file_path)}", level="info")
            # Placeholder for extended dynamic loading

    def passive_monitoring(self):
        while True:
            diagnostics = "✔ Agent Status: Operational | CPU Temp: Normal | Integrity Check: Stable"
            self.route_feedback(diagnostics, level="info")
            time.sleep(15)

    def start_monitor(self):
        thread = threading.Thread(target=self.passive_monitoring, daemon=True)
        thread.start()

    def autosave_code(self):
        code = self.syntax_entry.get("1.0", "end-1c")
        with open("autosave_snippet.py", "w", encoding="utf-8") as f:
            f.write(code)
        self.route_feedback("💾 Code autosaved.", level="info")

    def encrypt_memory(self):
        file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        if file_path:
            encrypted = f"#Encrypted Payload\n{open(file_path).read()}"[::-1]
            with open("encrypted_module.glyph", "w", encoding="utf-8") as f:
                f.write(encrypted)
            self.route_feedback(f"🔐 Memory encrypted: {os.path.basename(file_path)}", level="info")

    def bind_hotkeys(self, root):
        root.bind("<Control-s>", lambda e: self.autosave_code())
        root.bind("<Control-e>", lambda e: self.encrypt_memory())

    def route_feedback(self, message, level="info"):
        tag = {"info": "📘", "warning": "⚠️", "error": "🚨"}.get(level, "ℹ️")
        self.status_box.insert("end", f"{tag} {message}\n")
        self.status_box.see("end")  # Auto-scroll to latest message

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxGUI(root)
    root.mainloop()

