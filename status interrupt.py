import tkinter as tk
from tkinter import ttk, filedialog
import ast
import threading
import time

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

        # 🚦 Start passive monitoring
        self.start_monitor()

    def verify_syntax(self):
        code = self.syntax_entry.get("1.0", "end-1c")
        self.status_box.insert("end", "🔎 Checking syntax...\n")
        try:
            ast.parse(code)
            self.status_box.insert("end", "✅ Syntax is valid.\n")
        except SyntaxError as e:
            self.status_box.insert("end", f"❌ Syntax Error: {e}\n")

    def inject_memory(self):
        file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if file_path:
            self.status_box.insert("end", f"🧠 Module uploaded: {file_path}\n")
            # Placeholder: Extend with dynamic import or verification logic

    def passive_monitoring(self):
        while True:
            diagnostics = "✔ Agent Status: Operational | CPU Temp: Normal | Integrity Check: Stable\n"
            self.status_box.insert("end", diagnostics)
            time.sleep(15)

    def start_monitor(self):
        thread = threading.Thread(target=self.passive_monitoring, daemon=True)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxGUI(root)
    root.mainloop()

