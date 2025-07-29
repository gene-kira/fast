import tkinter as tk
from tkinter import ttk

def launch_guardian_gui():
    root = tk.Tk()
    root.title("MAGICBOX GUARDIAN v 4.5")
    root.geometry("1080x720")
    root.configure(bg="#101010")

    # Fonts
    title_font = ("Orbitron", 22)
    label_font = ("Consolas", 12)
    button_font = ("Consolas", 10)

    # Header
    tk.Label(root, text="MAGICBOX GUARDIAN v 4.5", font=title_font,
             fg="#FFA500", bg="#101010").place(x=30, y=20)

    tk.Label(root, text="ACTIVE DEFENSE MODE", font=label_font,
             fg="#00FF00", bg="#101010").place(x=780, y=25)

    # Persona Panel
    persona_frame = tk.LabelFrame(root, text="Persona", bg="#101010",
                                  fg="#FFA500", font=label_font)
    persona_frame.place(x=30, y=80, width=250, height=130)
    tk.Label(persona_frame, text="🧑 Baurdan", bg="#101010", fg="white", font=label_font).pack(anchor="w", padx=10, pady=10)
    tk.Button(persona_frame, text="MANUAL OVERRIDE", bg="#FFA500", fg="black", font=button_font).pack(padx=10, pady=5)

    # Logs Tabs
    log_tabs = ttk.Notebook(root)
    log_tabs.place(x=300, y=80, width=750, height=300)

    style = ttk.Style()
    style.theme_use("default")
    style.configure("TNotebook.Tab", background="#333333", foreground="#FFA500", padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", "#FFA500")], foreground=[("selected", "#101010")])

    def create_log_frame(log_texts):
        frame = tk.Frame(log_tabs, bg="#101010")
        log_box = tk.Text(frame, bg="#202020", fg="white", font=("Consolas", 10))
        for line in log_texts:
            log_box.insert("end", f"{line}\n")
        log_box.config(state="disabled")
        log_box.pack(fill="both", expand=True, padx=10, pady=10)
        return frame

    logs = [
        "Allowed Whitelisted IP 8.8.8.8",
        "Auto-blocked IP 203.0.113.42",
        "Unusual file activity detected.",
        "Badge Unlocked: Interceptor Mk II",
        "Allowed Manual IP 198.51.000.23",
        "Suspicious Packet: 10.0.0.5 - 192.168.1.2",
        "Auto-blocked IP 198.51.100.23",
        "Badge Unlocked: Shield Commander"
    ]

    log_tabs.add(create_log_frame(logs), text="LOGS")
    log_tabs.add(create_log_frame(["Live packet feed initialized...", "Monitoring port 443...", "Incoming packet flagged."]), text="LIVE PACKETS")
    log_tabs.add(create_log_frame(["VAULT breach attempt logged.", "Integrity check passed."]), text="VAULT EVENTS")

    # Threat Level
    tk.Label(root, text="THREAT LEVEL", font=label_font, fg="white", bg="#101010").place(x=30, y=230)
    ttk.Progressbar(root, value=30, maximum=100).place(x=150, y=238, width=120)

    # Badges
    badge_frame = tk.LabelFrame(root, text="Badges", bg="#101010",
                                fg="#FFA500", font=label_font)
    badge_frame.place(x=30, y=280, width=250, height=140)

    tk.Label(badge_frame, text="🔰 Interceptor Mk II", bg="#101010", fg="white", font=label_font).pack(anchor="w", padx=10, pady=5)
    tk.Label(badge_frame, text="🛡️ Shield Commander", bg="#101010", fg="white", font=label_font).pack(anchor="w", padx=10, pady=5)
    tk.Button(badge_frame, text="UNLOCK BADGE", bg="#FFA500", fg="black", font=button_font).pack(padx=10, pady=5)

    # Footer Buttons
    tk.Button(root, text="REFRESH", font=button_font, bg="#FFA500", fg="black").place(x=880, y=400, width=160, height=40)

    root.mainloop()

launch_guardian_gui()

