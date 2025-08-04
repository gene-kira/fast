# gui/overmind_gui.py
import tkinter as tk
import pyttsx3

engine = pyttsx3.init()
voice_active = False

def toggle_voice():
    global voice_active
    voice_active = not voice_active
    status = "Voice ON" if voice_active else "Voice OFF"
    print(f"[Voice] {status}")
    speak(status)

def speak(text):
    if voice_active:
        engine.say(text)
        engine.runAndWait()

def launch_gui():
    root = tk.Tk()
    root.title("Overmind AI Interface")
    root.geometry("600x400")
    root.configure(bg="black")

    label = tk.Label(root, text="OVERMIND ONLINE", font=("Courier", 24), fg="lime", bg="black")
    label.pack(pady=50)

    status = tk.Label(root, text="System Status: Nominal", font=("Courier", 16), fg="white", bg="black")
    status.pack()

    voice_button = tk.Button(root, text="Toggle Voice", command=toggle_voice, font=("Courier", 14), bg="gray", fg="white")
    voice_button.pack(pady=20)

    root.mainloop()

