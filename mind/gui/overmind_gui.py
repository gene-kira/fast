# gui/overmind_gui.py
import tkinter as tk

def launch_gui():
    root = tk.Tk()
    root.title("Overmind AI Interface")
    root.geometry("600x400")

    label = tk.Label(root, text="OVERMIND ONLINE", font=("Courier", 24), fg="lime")
    label.pack(pady=50)

    status = tk.Label(root, text="System Status: Nominal", font=("Courier", 16))
    status.pack()

    root.mainloop()

