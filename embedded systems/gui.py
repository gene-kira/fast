# gui.py
import tkinter as tk
from tkinter import ttk, messagebox

def launch_gui(dma, watchdog):
    root = tk.Tk()
    root.title("MagicBox GlyphGuard")
    root.geometry("850x500")
    root.configure(bg="#1E1E2F")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background="#1E1E2F", foreground="#DDEEFF", font=("Arial", 12))
    style.configure("TButton", padding=10, foreground="#00BFFF")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    bridge_tab = ttk.Frame(notebook)
    dma_tab = ttk.Frame(notebook)
    mem_tab = ttk.Frame(notebook)

    notebook.add(bridge_tab, text="Bridge Monitor")
    notebook.add(dma_tab, text="DMA Console")
    notebook.add(mem_tab, text="Memory Editor")

    # Status indicators
    status_vars = {
        "dma": tk.StringVar(value="Ready"),
        "last": tk.StringVar(value="None")
    }

    def update_status(msg):
        status_vars["last"].set(msg)

    def poll_status():
        status_vars["dma"].set("Busy" if dma.dma_busy else "Ready")
        root.after(1000, poll_status)

    poll_status()

    ttk.Label(bridge_tab, text="DMA Status:").pack(anchor="w")
    ttk.Label(bridge_tab, textvariable=status_vars["dma"]).pack(anchor="w")
    ttk.Label(bridge_tab, text="Last Event:").pack(anchor="w")
    ttk.Label(bridge_tab, textvariable=status_vars["last"]).pack(anchor="w")

    # DMA Console
    def start_dma():
        watchdog.refresh()
        msg = dma.trigger_dma(0x2000, 0x3000, 512)
        update_status(msg)

    ttk.Button(dma_tab, text="Start DMA Transfer", command=start_dma).pack(pady=10)

    # Memory Editor
    addr_entry = ttk.Entry(mem_tab)
    addr_entry.pack(pady=5)

    def read_mem():
        try:
            addr = int(addr_entry.get(), 16)
            val = dma.read_mem(addr)
            messagebox.showinfo("Memory Read", f"{hex(addr)} = {val}")
        except:
            messagebox.showerror("Error", "Bad Address")

    def write_mem():
        try:
            addr = int(addr_entry.get(), 16)
            dma.write_mem(addr, "42")
            messagebox.showinfo("Memory Write", f"Wrote 42 to {hex(addr)}")
        except:
            messagebox.showerror("Error", "Bad Address")

    ttk.Button(mem_tab, text="Read", command=read_mem).pack()
    ttk.Button(mem_tab, text="Write", command=write_mem).pack()

    root.mainloop()

