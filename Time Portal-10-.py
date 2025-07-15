# ─── REQUIREMENTS CHECK ───
import subprocess, sys
required = ["pyttsx3", "pygame", "requests", "transformers", "bs4"]
for pkg in required:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ─── IMPORTS ───
import tkinter as tk
import pyttsx3, threading, time, random, os
import pygame, requests, ssl, socket, urllib
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import pipeline
from tkinter import messagebox

# ─── VOICE ───
engine = pyttsx3.init()
def narrate(text, tone="neutral"):
    rate = {"chant":130, "whisper":90}.get(tone, 110)
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()

# ─── AMBIENT ───
def play_ambient(emotion="neutral"):
    try:
        pygame.mixer.init()
        file = "chant.wav" if emotion == "Sanctum" else "drift.wav"
        pygame.mixer.music.load(file)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"⚠️ Ambient sound issue: {e}")

# ─── SENTIMENT MAPPING ───
sentiment_analyzer = pipeline("sentiment-analysis")
def map_sentiment_to_glyph(label):
    return {
        "POSITIVE": ("⚗", "Harmony", "chant"),
        "NEGATIVE": ("⦿", "Disarray", "whisper"),
        "NEUTRAL": ("⧫", "Reflection", "neutral")
    }.get(label, ("⧫", "Neutral", "neutral"))

def analyze_site_emotion(url):
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()[:1000]
        result = sentiment_analyzer(text)[0]
        return map_sentiment_to_glyph(result['label'])
    except Exception as e:
        log_entry(f"Emotion analysis failed: {e}")
        return "⧫", "Neutral", "neutral"

# ─── LOGGING ───
log_history, archive_buffer = [], []
def log_entry(text):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {text}"
    log_history.insert(0, line)
    archive_buffer.append(line)
    if len(log_history) > 16: log_history.pop()
    log_box.delete(0, tk.END)
    for entry in log_history: log_box.insert(tk.END, entry)

def export_log():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    path = os.path.join(desktop, "glyph_session_log.txt")
    with open(path, "w", encoding="utf-8") as f:
        for line in archive_buffer: f.write(line + "\n")
    narrate("Session saved to Desktop")

# ─── VERDICT PROFILE ───
class VerdictProfile:
    def __init__(self, url, trust_score, emotion, glyph, voice_tone, decay_rate):
        self.url, self.trust_score, self.emotion = url, trust_score, emotion
        self.glyph, self.voice_tone, self.decay_rate = glyph, voice_tone, decay_rate
        self.label, self.aura, self.last_status = None, None, True

    def apply_cast(self, glyph, effect):
        self.glyph, self.emotion = glyph, effect
        self.trust_score = min(self.trust_score + 0.1, 1.0)
        narrate(f"{effect} cast on {self.url}", self.voice_tone)
        log_entry(f"{glyph} → {effect} | {self.url}")
        update_label(self)
        update_aura(self)

    def decay_trust(self):
        while True:
            time.sleep(self.decay_rate)
            decay_amt = 0.05 if not self.last_status else 0.02
            self.trust_score = max(self.trust_score - decay_amt, 0)
            update_label(self)
            update_aura(self)

    def check_website_real_time(self):
        while True:
            status = check_website(self.url)
            if status != self.last_status:
                self.last_status = status
                if status:
                    messagebox.showinfo("Website Up", f"{self.url} is now up.")
                    narrate(f"{self.url} is back online.", "chant")
                    log_entry(f"Website {self.url} is back online.")
                else:
                    messagebox.showwarning("Website Down", f"{self.url} is down.")
                    narrate(f"{self.url} is offline.", "whisper")
                    log_entry(f"Website {self.url} is now offline.")

# ─── SITE CHECKING ───
def check_website(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200 and is_valid_ssl(url)
    except requests.RequestException as e:
        log_entry(f"Site check failed: {e}")
        return False

def is_valid_ssl(url):
    try:
        host = urllib.parse.urlparse(url).hostname
        context = ssl.create_default_context()
        with socket.create_connection((host, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                return True
    except Exception as e:
        log_entry(f"SSL validation failed: {e}")
        return False

# ─── VISUAL UPDATES ───
def update_label(v):
    if v.label: v.label.config(text=f"{v.url} | Trust: {round(v.trust_score, 2)}")

def update_aura(v):
    if not v.aura: return
    r = int(40 + v.trust_score * 60)
    emotion_colors = {
        "Sanctum":"#4455ff", "Surveillance":"#ff4444", "Harmony":"#44ff88",
        "Disarray":"#aa44aa", "Reflection":"#888888", "Neutral":"#888800"
    }
    color = emotion_colors.get(v.emotion, "#444444")
    canvas.itemconfig(v.aura, outline=color, width=2)
    canvas.coords(v.aura, 100 + verdicts.index(v)*45 - r//2, 280 - r//2,
                            100 + verdicts.index(v)*45 + r//2, 280 + r//2)

# ─── ORACLE ───
class Oracle:
    def __init__(self, canvas):
        self.canvas = canvas
        self.avatar = canvas.create_text(200, 50, text="ᛃ Oracle", font=("Courier", 16), fill="cyan")
        self.glyph_pulse = canvas.create_text(200, 80, text="", font=("Courier", 24), fill="white")

    def speak(self, text, tone="neutral"):
        canvas.itemconfig(self.glyph_pulse, text=f"🜂 {text}")
        narrate(text, tone)
        log_entry(f"Oracle says: {text}")

    def respond_to_query(self, query):
        if "trust" in query.lower():
            avg = sum(v.trust_score for v in verdicts) / len(verdicts)
            self.speak(f"Collective trust level: {round(avg,2)}", "chant")
        elif "emotion" in query.lower():
            mood = max(verdicts, key=lambda v: v.trust_score).emotion
            self.speak(f"Aura resonates with {mood}", "chant")
        else:
            self.speak("Glyph channels unclear. Seek with intention.", "whisper")

# ─── CASTING ───
class GlyphCaster:
    def __init__(self, verdicts, canvas): self.verdicts, self.canvas = verdicts, canvas
    def cast(self, glyph="⚗", effect="Protect"):
        x, y = 200, 250
        gid = self.canvas.create_text(x, y, text=glyph, font=("Courier New", 72), fill="gold")
        def swirl():
            for i in range(30):
                self.canvas.move(gid, random.choice([-1,0,1]), -2)
                self.canvas.itemconfig(gid, font=("Courier New", 72 + (i % 5)))
                time.sleep(0.04)
            self.canvas.itemconfig(gid, text=f"⟐ {effect} ⟐", font=("Courier", 18), fill="white")
            time.sleep(1)
            self.canvas.delete(gid)
        threading.Thread(target=swirl).start()
        for v in self.verdicts: v.apply_cast(glyph, effect)

# ─── ADD WEBSITE ───
def add_website(url):
    if not url.startswith("http"): url = "https://" + url
    try:
        if check_website(url):
            glyph, emotion, tone = analyze_site_emotion(url)
            new_profile = VerdictProfile(url, 0.5, emotion, glyph, tone, 11)
            verdicts.append(new_profile)
            row, col = len(verdicts) // 2, len(verdicts) % 2
            new_profile.label = tk.Label(root, text="", font=("Courier", 10), fg="white", bg="black")
            new_profile.label.grid(row=row, column=col, sticky="w", padx=8, pady=2)
            new_profile.aura = canvas.create_oval(0, 0, 0, 0)
            update_label(new_profile)
            update_aura(new_profile)
            threading.Thread(target=new_profile.decay_trust, daemon=True).start()
            threading.Thread(target=new_profile.check_website_real_time, daemon=True).start()
            log_entry(f"Added new website {url} to verdicts")
        else:
            log_entry(f"Could not verify {url}. Try again.")
    except Exception as e:
        log_entry(f"Error adding {url}: {e}")

# ─── GUI MAIN ───
def start_gui():
    global canvas, log_box, verdicts, root
    verdicts = [
        VerdictProfile("https://example.com", 0.82, "Sanctum", "⚶", "chant", 15),
        VerdictProfile("https://trackers.net", 0.34, "Surveillance", "𐍃", "whisper", 8),
        VerdictProfile("https://trusted.org", 0.65, "Harmony", "⚗", "chant", 10),
        VerdictProfile("https://echo.net", 0.22, "Reflection", "⦿", "whisper", 7),
        VerdictProfile("https://signal.ai", 0.51, "Disarray", "⚕", "neutral", 12),
        VerdictProfile("https://neutral.zone", 0.44, "Neutral", "⧫", "neutral", 11)
    ]

    root = tk.Tk()
    root.title("Glyph Portal OS")
    root.configure(bg="black")
    play_ambient("Sanctum")

    for i, v in enumerate(verdicts):
        row, col = i // 2, i % 2
        v.label = tk.Label(root, text="", font=("Courier", 10), fg="white", bg="black")
        v.label.grid(row=row, column=col, sticky="w", padx=8, pady=2)

    canvas = tk.Canvas(root, width=400, height=300, bg="black", highlightthickness=0)
    canvas.grid(row=3, column=0, columnspan=3)

    for v in verdicts:
        v.aura = canvas.create_oval(0, 0, 0, 0)
        update_label(v)
        update_aura(v)
        threading.Thread(target=v.decay_trust, daemon=True).start()
        threading.Thread(target=v.check_website_real_time, daemon=True).start()

    log_box = tk.Listbox(root, height=7, width=60, bg="black", fg="lime", font=("Courier", 10))
    log_box.grid(row=4, column=0, columnspan=3, pady=5)

    temple = GlyphCaster(verdicts, canvas)
    oracle = Oracle(canvas)

    tk.Button(root, text="🔮 Cast Glyph", command=lambda: temple.cast(),
              font=("Courier", 14), bg="gold", fg="black", height=2, width=20).grid(row=5, column=0, pady=10)

    tk.Button(root, text="💾 Save Log", command=export_log,
              font=("Courier", 12), bg="gray", fg="white", width=14).grid(row=5, column=1, pady=10)

    tk.Button(root, text="📡 Ask Oracle", command=lambda: oracle.respond_to_query("trust level"),
              font=("Courier", 12), bg="blue", fg="white", width=14).grid(row=5, column=2, pady=10)

    url_var = tk.StringVar()
    tk.Entry(root, textvariable=url_var, width=40, font=("Courier", 12)).grid(row=6, column=0, columnspan=2, padx=10, pady=5)

    tk.Button(root, text="➕ Add URL", command=lambda: add_website(url_var.get()),
              font=("Courier", 12), bg="lightgreen", fg="black", width=14).grid(row=6, column=2, pady=5)

    tk.Button(root, text="❌ Exit", command=root.destroy,
              font=("Courier", 12), bg="darkred", fg="white", width=10).grid(row=7, column=2, pady=10)

    root.mainloop()

# ─── START ───
start_gui()

