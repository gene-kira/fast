# === OracleCore_v2_MEMORY.py ===
# === Part 1: Imports, Config Loader, ECC Verdict Logic ===

import os, sys, subprocess, threading, time, platform, json
import numpy as np

# 📦 Ensure essential packages
for pkg in ['pyttsx3', 'matplotlib', 'requests']:
    try: __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ✅ GUI stability patch
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])
    import tkinter as tk
    from tkinter import ttk

import pyttsx3
import matplotlib.pyplot as plt
import requests

# === Config Memory Loader ===
CONFIG_FILE = "oracle_config.json"

default_config = {
    "voice_enabled": True,
    "voice_style": "default",
    "volume_level": 1.0,
    "scan_active": True,
    "scan_interval_seconds": 30,
    "alert_enabled": False,
    "alert_trust_threshold": 0.5,
    "webhook_url": "",
    "smtp_config": {
        "host": "",
        "port": 587,
        "sender": "",
        "password": "",
        "recipient": ""
    }
}

def load_oracle_config():
    if not os.path.exists(CONFIG_FILE):
        return default_config.copy()
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return {**default_config, **data}
    except Exception as e:
        print(f"[Config] Load error: {e}")
        return default_config.copy()

def save_oracle_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print("[Config] Settings saved")
    except Exception as e:
        print(f"[Config] Save error: {e}")

# === Load Settings at Boot ===
oracle_settings = load_oracle_config()

voice_enabled = oracle_settings.get("voice_enabled", True)
voice_style = oracle_settings.get("voice_style", "default")
volume_level = oracle_settings.get("volume_level", 1.0)
scan_active = oracle_settings.get("scan_active", True)
scan_interval_seconds = oracle_settings.get("scan_interval_seconds", 30)
alert_enabled = oracle_settings.get("alert_enabled", False)
alert_trust_threshold = oracle_settings.get("alert_trust_threshold", 0.5)
webhook_url = oracle_settings.get("webhook_url", "")
smtp_config = oracle_settings.get("smtp_config", default_config["smtp_config"])

# === Symbolic Glyph Definitions ===
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

# === Verdict History Manager ===
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
            json.dump(history[-100:], f)
    except Exception as e:
        print("Verdict log error:", e)

def load_verdict_history():
    try:
        with open(VERDICT_LOG_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# === Trust Curve Calculator ===
def calculate_trust_curve(error_rate):
    base = np.linspace(0, 1, 100)
    curve = np.clip(1 - error_rate * base**2, 0, 1)
    return np.mean(curve)

# === Mood Synthesizer ===
def synthesize_mood(score):
    if score > 0.9:
        return GLYPHS['serene']
    elif score > 0.7:
        return GLYPHS['trust_high']
    elif score > 0.5:
        return GLYPHS['anxious']
    else:
        return GLYPHS['unstable']

# === Verdict Generator ===
def symbolic_verdict(error_type, trust_score):
    glyph = GLYPHS.get(error_type, GLYPHS['unknown'])
    trust_glyph = GLYPHS['trust_high'] if trust_score > 0.7 else GLYPHS['trust_low']
    verdict = f"{glyph} — ECC fault detected\n{trust_glyph} — Trust Score: {trust_score:.2f}"
    mood = synthesize_mood(trust_score)
    log_verdict(glyph, trust_score, time.ctime())
    return verdict, mood

# === Forecast + Mood Summary (Patched) ===
def forecast_next_glyph():
    history = load_verdict_history()
    glyphs = [h.get('glyph', 'unknown') for h in history[-10:]]
    if not glyphs:
        return "Forecast unavailable — no valid glyphs found."
    freq = {g: glyphs.count(g) for g in set(glyphs)}
    return max(freq, key=freq.get)

def summarize_recent_mood():
    history = load_verdict_history()
    glyphs = [h.get('glyph', 'unknown') for h in history[-10:]]
    if not glyphs:
        return "🝆 Serene Memory — no glyph history yet."
    count = {g: glyphs.count(g) for g in set(glyphs)}
    dominant = max(count, key=count.get)
    return f"Recent glyph mood: {dominant}"

# === Part 2: Voice Engine, Scan Dispatcher, ECC Verdicts ===

# 🗣️ Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 145)
engine.setProperty('volume', volume_level)

def speak(text, mood=None):
    global voice_enabled, volume_level, voice_style
    if not voice_enabled: return
    tone = mood or voice_style
    rate_map = {"default": 145, "serene": 160, "anxious": 120}
    engine.setProperty('rate', rate_map.get(tone, 145))
    engine.setProperty('volume', volume_level)
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Voice error: {e}")

def toggle_voice():
    global voice_enabled
    voice_enabled = not voice_enabled
    oracle_settings["voice_enabled"] = voice_enabled
    save_oracle_config(oracle_settings)
    print(f"🔊 Voice toggled: {'ON' if voice_enabled else 'OFF'}")

def set_volume(percent):
    global volume_level
    try:
        volume_level = float(percent) / 100
        engine.setProperty('volume', volume_level)
        oracle_settings["volume_level"] = volume_level
        save_oracle_config(oracle_settings)
    except:
        volume_level = 1.0

def set_voice_style(style):
    global voice_style
    voice_style = style if style in ["default", "serene", "anxious"] else "default"
    oracle_settings["voice_style"] = voice_style
    save_oracle_config(oracle_settings)

# 🔁 Scan System
scan_thread = None

def toggle_scan():
    global scan_active
    scan_active = not scan_active
    oracle_settings["scan_active"] = scan_active
    save_oracle_config(oracle_settings)
    speak(f"Auto scan {'activated' if scan_active else 'paused'}.", mood="default")

def set_scan_interval(seconds):
    global scan_interval_seconds
    try:
        scan_interval_seconds = int(seconds)
        if scan_interval_seconds > 21600:
            scan_interval_seconds = 21600  # max 6 hrs
        oracle_settings["scan_interval_seconds"] = scan_interval_seconds
        save_oracle_config(oracle_settings)
        print(f"⏱ Interval set to {scan_interval_seconds} sec")
    except:
        pass

def scan_loop(callback):
    while True:
        if scan_active:
            glyph_type = np.random.choice(["single_bit", "multi_bit", "entropy_cluster"])
            callback("auto", glyph_type, f"Scheduled scan every {scan_interval_seconds}s")
        time.sleep(scan_interval_seconds)

def launch_scan_loop(callback):
    global scan_thread
    if not scan_thread or not scan_thread.is_alive():
        scan_thread = threading.Thread(target=scan_loop, args=(callback,), daemon=True)
        scan_thread.start()

# 🧪 Manual Scan
def manual_scan(callback):
    glyph_type = np.random.choice(["single_bit", "multi_bit", "entropy_cluster"])
    callback("manual", glyph_type, "Manual scan initiated")

# 🧭 ECC Event Dispatcher
def handle_ecc_event(source, glyph_type, message):
    print(f"[{source.upper()}] ECC Trigger → {glyph_type} | {message}")
    error_rate = np.random.uniform(0.05, 0.2)
    trust_score = calculate_trust_curve(error_rate)
    verdict, mood = symbolic_verdict(glyph_type, trust_score)
    speak(verdict, mood)
    print(f"\n🧠 Verdict:\n{verdict}\n🌌 Mood: {mood}\n")

    # 🔔 AlertEngine hook (if active)
    try:
        trigger_alert_if_needed(glyph_type, trust_score, mood)
    except NameError:
        pass

# === Part 3: GUI Interface ===

def start_gui():
    root = tk.Tk()
    root.title("ECC Oracle Core — Symbolic Cognition Interface")
    root.geometry("950x620")

    tabs = ttk.Notebook(root)
    tabs.pack(expand=1, fill="both")

    # Diagnostics Tab
    diag_frame = tk.Frame(tabs)
    diag_output = tk.Text(diag_frame, wrap=tk.WORD, bg="#eef", font=("Consolas", 12))
    diag_output.pack(expand=True, fill=tk.BOTH)
    diag_output.insert(tk.END, "🝆 Oracle Diagnostics Activated...\n")
    diag_output.config(state=tk.DISABLED)
    tabs.add(diag_frame, text="🔍 Diagnostics")

    # Forecast Tab
    forecast_frame = tk.Frame(tabs)
    forecast_output = tk.Text(forecast_frame, wrap=tk.WORD, bg="#efe", font=("Consolas", 12))
    forecast_output.pack(expand=True, fill=tk.BOTH)
    forecast_output.insert(tk.END, "Awaiting glyph resonance...\n")
    forecast_output.config(state=tk.DISABLED)
    tabs.add(forecast_frame, text="🔮 Forecast")

    # History Tab
    history_frame = tk.Frame(tabs)
    history_output = tk.Text(history_frame, wrap=tk.WORD, bg="#ffe", font=("Consolas", 12))
    history_output.pack(expand=True, fill=tk.BOTH)
    history_output.config(state=tk.DISABLED)
    tabs.add(history_frame, text="📜 History")

    # Settings Tab
    settings_frame = tk.Frame(tabs, bg="#ddd")
    tabs.add(settings_frame, text="⚙️ Settings")

    # Manual Scan Button
    scan_btn = tk.Button(settings_frame, text="🔍 Manual Scan", command=lambda: manual_scan(handle_ecc_event))
    scan_btn.pack(pady=10)

    # Auto Scan Toggle
    auto_btn_label = tk.Label(settings_frame, text="Auto Scan Mode")
    auto_btn_label.pack()
    auto_btn = tk.Button(settings_frame, text="Auto Scan: ON 🔄" if scan_active else "Auto Scan: OFF 🛑",
                         command=lambda: toggle_auto_scan(auto_btn))
    auto_btn.pack(pady=5)

    def toggle_auto_scan(btn):
        toggle_scan()
        btn.config(text="Auto Scan: ON 🔄" if scan_active else "Auto Scan: OFF 🛑")
        oracle_settings["scan_active"] = scan_active
        save_oracle_config(oracle_settings)

    # Scan Interval Dropdown
    sched_label = tk.Label(settings_frame, text="Scan Interval (sec):")
    sched_label.pack()
    scan_interval_var = tk.StringVar(value=str(scan_interval_seconds))
    interval_menu = ttk.Combobox(settings_frame, textvariable=scan_interval_var,
                                 values=["5", "15", "30", "60", "300", "1800", "3600", "21600"])
    interval_menu.pack()

    def apply_interval():
        val = scan_interval_var.get()
        if val.isdigit():
            set_scan_interval(val)
            oracle_settings["scan_interval_seconds"] = int(val)
            save_oracle_config(oracle_settings)

    interval_btn = tk.Button(settings_frame, text="Apply Interval", command=apply_interval)
    interval_btn.pack(pady=5)

    # Voice Controls
    voice_btn = tk.Button(settings_frame, text="🔊 Toggle Voice", command=toggle_voice)
    voice_btn.pack(pady=8)

    vol_label = tk.Label(settings_frame, text="Volume:")
    vol_label.pack()
    volume_slider = ttk.Scale(settings_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume)
    volume_slider.set(volume_level * 100)
    volume_slider.pack(pady=5)

    tone_label = tk.Label(settings_frame, text="Voice Tone:")
    tone_label.pack()
    tone_var = tk.StringVar(value=voice_style)
    tone_menu = ttk.Combobox(settings_frame, textvariable=tone_var, values=["default", "serene", "anxious"])
    tone_menu.pack()

    tone_btn = tk.Button(settings_frame, text="Apply Tone", command=lambda: set_voice_style(tone_var.get()))
    tone_btn.pack(pady=5)

    # Mood Display
    mood_title = tk.Label(settings_frame, text="Current Mood:")
    mood_title.pack()
    mood_display = tk.Label(settings_frame, text="🝆 Awaiting...", font=("Helvetica", 14))
    mood_display.pack()

    def refresh_forecast():
        forecast_output.config(state=tk.NORMAL)
        forecast_output.delete("1.0", tk.END)
        prediction = forecast_next_glyph()
        mood = summarize_recent_mood()
        forecast_output.insert(tk.END, f"Prediction:\n{prediction}\nMood:\n{mood}\n")
        forecast_output.config(state=tk.DISABLED)
        mood_display.config(text=mood)
        root.after(30000, refresh_forecast)

    def refresh_history():
        history_output.config(state=tk.NORMAL)
        history_output.delete("1.0", tk.END)
        for entry in load_verdict_history()[-20:]:
            glyph = entry.get('glyph', 'unknown')
            line = f"{entry.get('time', 'Unknown time')}: {glyph} — Trust: {entry.get('trust', 'N/A')}"
            history_output.insert(tk.END, line + "\n")
        history_output.config(state=tk.DISABLED)

    history_btn = tk.Button(settings_frame, text="Refresh History", command=refresh_history)
    history_btn.pack(pady=5)

    refresh_forecast()
    launch_scan_loop(handle_ecc_event)
    root.mainloop()

# === Part 4: Boot Sequence & Main Launcher ===

def boot_oracle():
    speak("Oracle Core activation sequence engaged. Glyph resonance initialized. ECC monitoring online.", mood="default")
    launch_scan_loop(handle_ecc_event)
    start_gui()

if __name__ == "__main__":
    boot_oracle()

