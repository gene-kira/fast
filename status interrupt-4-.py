import tkinter as tk
from tkinter import ttk, filedialog
import ast
import threading
import time
import os
import subprocess

class MagicBoxGUI:
    def __init__(self, root):
        root.title("MagicBox GlyphGuard — ASI Agent Panel")
        root.geometry("1000x700")
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
        ttk.Button(syntax_frame, text="Verify Syntax", command=self.verify_syntax).pack(pady=5)
        ttk.Button(syntax_frame, text="Run Sandbox", command=self.execute_code_sandbox).pack(pady=5)

        # 🧠 Memory Augmentation Tab
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Memory Augmentation")

        ttk.Label(memory_frame, text="Upload Memory Module:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        ttk.Button(memory_frame, text="Inject Memory", command=self.inject_memory).pack(pady=5)
        ttk.Button(memory_frame, text="Encrypt Memory", command=self.encrypt_memory).pack(pady=5)

        # 🗝️ Registry Node Tab
        registry_frame = ttk.Frame(notebook)
        notebook.add(registry_frame, text="Registry Nodes")
        self.registry_map = {}

        ttk.Label(registry_frame, text="Set Agent Key/Value:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        self.key_entry = tk.Entry(registry_frame, bg="#2a2a2a", fg="#ffffff")
        self.key_entry.pack(padx=20, pady=2, fill="x")
        self.value_entry = tk.Entry(registry_frame, bg="#2a2a2a", fg="#ffffff")
        self.value_entry.pack(padx=20, pady=2, fill="x")
        ttk.Button(registry_frame, text="Encrypt & Register", command=self.register_encrypted_node).pack(pady=5)
        self.registry_display = tk.Text(registry_frame, height=8, bg="#2a2a2a", fg="#00ffff")
        self.registry_display.pack(padx=20, pady=5, fill="both")

        # 🧭 Visual Memory Map Tab
        visual_frame = ttk.Frame(notebook)
        notebook.add(visual_frame, text="Memory Map")
        self.memory_status = tk.Label(visual_frame, text="🗺️ Loaded modules: None", foreground="#00ffff", background="#1c1c1c")
        self.memory_status.pack(pady=20)

        # ⚙️ Agent Status Tab
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="Agent Status")
        ttk.Label(status_frame, text="System Diagnostics:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        self.status_box = tk.Text(status_frame, height=10, bg="#2a2a2a", fg="#00ff00", insertbackground="#ffffff")
        self.status_box.pack(padx=20, pady=5, fill="both")

        # 💻 Command Console Tab
        self.add_console_tab(notebook)

        # 🧬 Agent Script Overlay Tab
        script_frame = ttk.Frame(notebook)
        notebook.add(script_frame, text="Agent Overlay")
        ttk.Label(script_frame, text="Agent Behavior Script:", foreground="#ffffff", background="#1c1c1c").pack(pady=10)
        self.script_entry = tk.Text(script_frame, height=10, bg="#2a2a2a", fg="#ffffff")
        self.script_entry.pack(padx=20, pady=5, fill="both")
        ttk.Button(script_frame, text="Execute Overlay", command=self.run_agent_script).pack(pady=5)

        # Startup routines
        self.bind_hotkeys(root)
        self.start_monitor()

  def run_console_command(self):
        cmd = self.console_entry.get()
        try:
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
            self.console_output.insert("end", f"> {cmd}\n{result}\n")
        except subprocess.CalledProcessError as e:
            self.console_output.insert("end", f"⚠️ Error:\n{e.output}\n")
        self.console_output.see("end")

    def route_feedback(self, message, level="info"):
        tag = {"info": "📘", "warning": "⚠️", "error": "🚨"}.get(level, "ℹ️")
        self.status_box.insert("end", f"{tag} {message}\n")
        self.status_box.see("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicBoxGUI(root)
    root.mainloop()

