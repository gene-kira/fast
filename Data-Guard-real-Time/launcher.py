import tkinter as tk
from watchdog import run_watchdog
from PIL import ImageTk, Image
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def start_guard():
    speak("System Data Guard online.")
    status_label.config(text="✅ Guard is active")
    run_watchdog()

app = tk.Tk()
app.title("🛡️ MagicBox Guardian")
app.geometry("400x350")
app.configure(bg="#101820")

# Load holographic badge
badge_img = Image.open("resources/badge.png").resize((180, 180))
badge_tk = ImageTk.PhotoImage(badge_img)
badge = tk.Label(app, image=badge_tk, bg="#101820")
badge.pack(pady=10)

# Activation button
btn = tk.Button(app, text="ACTIVATE GUARD", command=start_guard,
                bg="#00c2ff", fg="white", font=("Verdana", 16, "bold"))
btn.pack()

# Status label
status_label = tk.Label(app, text="🕑 Standby", fg="#cfcfcf",
                        bg="#101820", font=("Verdana", 14))
status_label.pack(pady=10)

app.mainloop()

