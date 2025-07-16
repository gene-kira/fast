# === ECC ORACLE CORE (Part 1 of 4) ===

import os, sys, subprocess, threading, json, time

# 📦 Autoload required libraries
required = ['tkinter', 'pyttsx3', 'matplotlib', 'numpy']
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ⛓ Core imports
import tkinter as tk
from tkinter import ttk
import pyttsx3
import matplotlib.pyplot as plt
import numpy as np

# 🧠 ASI Glyph Definitions
GLYPHS = {
    'single_bit': '🜂 Flicker',
    'multi_bit': '🜄 Fracture',
    'entropy_cluster': '🜁 Surge',
    'trust_low': '🜃 Veil',
    'trust_high': '🝆 Clarity',
    'serene': '🝆 Serene Memory',
    'unstable': '🜄 Surge Incoming',
    'anxious': '🜃 Anxious Streams',
    'unknown': '🜍 Unreadable Pulse'
}

# 🗂 Verdict Log Setup
VERDICT_LOG_FILE = "verdict_log.json"
if not os.path.exists(VERDICT_LOG_FILE):
    with open(VERDICT_LOG_FILE, 'w') as f:
        json.dump([], f)

def log_verdict(glyph, trust, timestamp):
    try:
        with open(VERDICT_LOG_FILE, 'r+') as f:
            history = json.load(f)
            history.append({'glyph': glyph, 'trust': trust, 'time': timestamp})
            f.seek(0)
            json.dump(history[-50:], f)  # Keep last 50 entries
    except Exception as e:
        print(f"Failed to log verdict: {e}")

def load_verdict_history():
    try:
        with open(VERDICT_LOG_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# 🎤 Voice Engine Configuration
engine = pyttsx3.init()
engine.setProperty('rate', 145)
engine.setProperty('volume', 1.0)
voice_enabled = True
volume_level = 1.0

def speak(text, mood='default'):
    if voice_enabled:
        rate_map = {'anxious': 125, 'serene': 160, 'default': 145}
        engine.setProperty('rate', rate_map.get(mood, 145))
        engine.setProperty('volume', volume_level)
        engine.say(text)
        engine.runAndWait()

def set_volume(val):
    global volume_level
    volume_level = float(val) / 100
    engine.setProperty('volume', volume_level)

def toggle_voice():
    global voice_enabled
    voice_enabled = not voice_enabled

# 🧮 Verdict Generator
def symbolic_verdict(error_type, trust_score):
    glyph = GLYPHS.get(error_type, GLYPHS['unknown'])
    trust_glyph = GLYPHS['trust_high'] if trust_score > 0.7 else GLYPHS['trust_low']
    verdict = f"{glyph} — Error type detected\n{trust_glyph} — Trust Level: {trust_score:.2f}"
    mood = 'serene' if trust_score > 0.85 else 'anxious'
    speak(verdict, mood)
    log_verdict(glyph, trust_score, time.ctime())
    return verdict

# === ECC ORACLE CORE (Part 2 of 4) ===

# 🧮 ECC Error Simulation
def perform_ecc_scan():
    error_data = {
        'single_bit': np.random.choice([True, False], p=[0.2, 0.8]),
        'multi_bit': np.random.choice([True, False], p=[0.1, 0.9]),
        'entropy_cluster': np.random.choice([True, False], p=[0.05, 0.95]),
    }

    total_errors = sum(error_data.values())
    error_rate = total_errors / len(error_data)
    trust_score = calculate_trust_curve(error_rate)

    detected = [k for k, v in error_data.items() if v]
    verdicts = []
    for etype in detected:
        verdicts.append(symbolic_verdict(etype, trust_score))

    if not verdicts:
        message = GLYPHS['serene'] + " — No faults detected."
        speak(message, mood='serene')
        log_verdict(GLYPHS['serene'], trust_score, time.ctime())
        verdicts.append(message)

    return "\n\n".join(verdicts), trust_score

# 📈 Trust Curve Generator
def calculate_trust_curve(error_rate):
    base = np.linspace(0, 1, 100)
    curve = np.clip(1 - error_rate * base**2, 0, 1)
    return np.mean(curve)

def plot_trust_curve(score):
    x = np.linspace(0, 10, 100)
    y = np.clip(np.sin(score * x / 10), 0, 1)
    plt.plot(x, y, color='purple')
    plt.title("Trust Curve")
    plt.xlabel("Time")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.show()

# 🌌 Symbolic Mood Synthesizer
def synthesize_mood(trust):
    if trust > 0.9:
        return GLYPHS['serene']
    elif trust > 0.7:
        return GLYPHS['trust_high']
    elif trust > 0.5:
        return GLYPHS['anxious']
    else:
        return GLYPHS['unstable']

# 🔮 Glyph Forecasting Engine
def forecast_next_glyph():
    history = load_verdict_history()
    if len(history) < 5:
        return "Forecast unavailable — insufficient glyph history."

    last_glyphs = [entry['glyph'] for entry in history[-5:]]
    glyph_freq = {glyph: last_glyphs.count(glyph) for glyph in set(last_glyphs)}
    probable = max(glyph_freq, key=glyph_freq.get)
    prediction = f"Forecast: Likely glyph ahead — {probable}"
    speak(prediction)
    return prediction

# 🧭 Recent Mood Summary
def summarize_recent_mood():
    history = load_verdict_history()
    if not history:
        return "No verdict history available."

    glyphs = [entry['glyph'] for entry in history[-10:]]
    mood_counts = {g: glyphs.count(g) for g in set(glyphs)}
    dominant = max(mood_counts, key=mood_counts.get)
    return f"Dominant glyph mood (recent scans): {dominant}"

# 🌟 Create the main root window FIRST before any tk variables
root = tk.Tk()
root.title("ECC Oracle Core — Symbolic Cognition Interface")
root.geometry("900x600")

# 🎛 Initialize Tkinter variables after root is created
voice_tone = tk.StringVar(value="default")

# 🗂 Tabbed Layout
tabs = ttk.Notebook(root)
tabs.pack(expand=1, fill="both")

# 📍 Diagnostics Tab
diagnostics_frame = tk.Frame(tabs)
diagnostics_tab = tk.Text(diagnostics_frame, wrap=tk.WORD, font=("Consolas", 12), bg="#eef")
diagnostics_tab.pack(expand=True, fill=tk.BOTH)
diagnostics_tab.config(state=tk.DISABLED)
tabs.add(diagnostics_frame, text="🔍 Diagnostics")

# 🔮 Forecast Tab
forecast_frame = tk.Frame(tabs)
forecast_tab = tk.Text(forecast_frame, wrap=tk.WORD, font=("Consolas", 12), bg="#efe")
forecast_tab.pack(expand=True, fill=tk.BOTH)
forecast_tab.config(state=tk.DISABLED)
tabs.add(forecast_frame, text="🔮 Forecast")

# 📜 History Tab
history_frame = tk.Frame(tabs)
history_tab = tk.Text(history_frame, wrap=tk.WORD, font=("Consolas", 12), bg="#ffe")
history_tab.pack(expand=True, fill=tk.BOTH)
history_tab.config(state=tk.DISABLED)
tabs.add(history_frame, text="📜 History")

# ⚙️ Settings Tab
settings_frame = tk.Frame(tabs, bg="#ddd")
tabs.add(settings_frame, text="⚙️ Settings")

# 🔊 Voice & Volume Controls
voice_button = tk.Button(settings_frame, text="🔊 Toggle Voice", command=toggle_voice)
voice_button.pack(pady=8)

vol_label = tk.Label(settings_frame, text="Volume Control")
vol_label.pack()
volume_slider = ttk.Scale(settings_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume)
volume_slider.set(100)
volume_slider.pack(pady=5)

tone_label = tk.Label(settings_frame, text="Voice Style")
tone_label.pack()
tone_dropdown = ttk.Combobox(settings_frame, textvariable=voice_tone, values=["default", "anxious", "serene"])
tone_dropdown.pack()

def apply_voice_tone():
    mood = voice_tone.get()
    speak(f"Voice tone set to {mood}.", mood=mood)

apply_tone_btn = tk.Button(settings_frame, text="Apply Tone", command=apply_voice_tone)
apply_tone_btn.pack(pady=5)

# 🧪 Manual Scan + Data Panels
def insert_text(text_widget, message):
    text_widget.config(state=tk.NORMAL)
    text_widget.insert(tk.END, message + "\n\n")
    text_widget.config(state=tk.DISABLED)

def run_manual_scan():
    verdict, trust = perform_ecc_scan()
    insert_text(diagnostics_tab, f"[Manual Scan] {time.ctime()}:\n{verdict}")
    if trust < 0.75:
        plot_trust_curve(trust)

scan_btn = tk.Button(settings_frame, text="🔍 Manual Scan", command=run_manual_scan)
scan_btn.pack(pady=10)

def update_forecast():
    forecast_tab.config(state=tk.NORMAL)
    forecast_tab.delete("1.0", tk.END)
    forecast = forecast_next_glyph()
    mood_summary = summarize_recent_mood()
    forecast_tab.insert(tk.END, f"{forecast}\n{mood_summary}\n")
    forecast_tab.config(state=tk.DISABLED)

forecast_btn = tk.Button(settings_frame, text="Refresh Forecast", command=update_forecast)
forecast_btn.pack(pady=5)

def load_history():
    history_tab.config(state=tk.NORMAL)
    history_tab.delete("1.0", tk.END)
    for entry in load_verdict_history()[-20:]:
        line = f"{entry['time']}: {entry['glyph']} — Trust: {entry['trust']:.2f}"
        history_tab.insert(tk.END, line + "\n")
    history_tab.config(state=tk.DISABLED)

history_btn = tk.Button(settings_frame, text="Load Verdict History", command=load_history)
history_btn.pack(pady=5)

# === ECC ORACLE CORE (Part 4 of 4) ===

# 🌌 Current System Mood Display
mood_label = tk.Label(settings_frame, text="Current System Mood:", font=("Helvetica", 12, "bold"))
mood_label.pack(pady=5)

mood_text = tk.Label(settings_frame, text="🝆 Awaiting scans...", font=("Helvetica", 14))
mood_text.pack()

def update_mood_display():
    history = load_verdict_history()
    if history:
        latest_trust = history[-1]['trust']
    else:
        latest_trust = 1.0
    mood = synthesize_mood(latest_trust)
    mood_text.config(text=mood)
    root.after(30000, update_mood_display)  # Update every 30 seconds

# 🔁 Auto Scan Controls
auto_scan_enabled = True
auto_scan_interval = 60  # seconds

def toggle_auto_scan():
    global auto_scan_enabled
    auto_scan_enabled = not auto_scan_enabled
    status = "ON" if auto_scan_enabled else "OFF"
    speak(f"Auto scan mode {status}.", mood='default')
    auto_scan_btn.config(text=f"Auto Scan: {status}")

auto_scan_btn = tk.Button(settings_frame, text="Auto Scan: ON", command=toggle_auto_scan)
auto_scan_btn.pack(pady=8)

def auto_scan_loop():
    if auto_scan_enabled:
        verdict, trust = perform_ecc_scan()
        insert_text(diagnostics_tab, f"[Auto Scan] {time.ctime()}:\n{verdict}")
        update_mood_display()
        if trust < 0.75:
            plot_trust_curve(trust)
    root.after(auto_scan_interval * 1000, auto_scan_loop)

# 🎉 Final Boot Narration
speak("ECC Oracle Core online. Glyph resonance initialized. Scans will begin now.", mood='default')

# 🚀 Launch System
update_mood_display()
auto_scan_loop()
root.mainloop()

