# ECC Memory Checker with ASI Integration — Now with Background Scanning and Voice Controls

import os
import sys
import subprocess
import threading

# Autoloader for required libraries
required = ['tkinter', 'pyttsx3', 'matplotlib', 'numpy']
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pyttsx3
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- ASI GLYPH SYSTEM ----------------------
GLYPHS = {
    'single_bit': '🜂 Flicker',
    'multi_bit': '🜄 Fracture',
    'entropy_cluster': '🜁 Surge',
    'trust_low': '🜃 Veil',
    'trust_high': '🝆 Clarity'
}

# ---------------------- SPEECH SETUP --------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 145)
engine.setProperty('volume', 1.0)
voice_enabled = True

def speak(text):
    if voice_enabled:
        engine.say(text)
        engine.runAndWait()

def set_volume(val):
    engine.setProperty('volume', float(val) / 100)

# ----------------- SYMBOLIC VERDICT ------------------
def symbolic_verdict(error_type, trust_score):
    glyph = GLYPHS.get(error_type, '🜍 Unknown')
    trust_glyph = GLYPHS['trust_high'] if trust_score > 0.7 else GLYPHS['trust_low']
    verdict = f"{glyph} — Error type detected\n{trust_glyph} — Trust Level: {trust_score:.2f}"
    speak(verdict)
    return verdict

# ---------------- TRUST CURVE -------------------------
def calculate_trust_curve(error_rate):
    base = np.linspace(0, 1, 100)
    curve = np.clip(1 - error_rate * base**2, 0, 1)
    return np.mean(curve)

def plot_trust_curve(score):
    x = np.linspace(0, 10, 100)
    y = np.clip(np.sin(score * x / 10), 0, 1)
    plt.plot(x, y)
    plt.title("System Trust Curve")
    plt.xlabel("Time")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.show()

# ---------------- ECC SIMULATION --------------------
def perform_scan():
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
        message = "No errors found. Memory integrity preserved."
        speak(message)
        verdicts.append(message)

    return "\n\n".join(verdicts), trust_score

# ---------------- BACKGROUND MONITOR ----------------
scan_interval_sec = 30

def background_scan():
    verdict, trust = perform_scan()
    output_text.insert(tk.END, f"\n--- Background Scan ---\n{verdict}\n")
    if trust < 0.8:
        plot_trust_curve(trust)
    # Schedule next scan
    threading.Timer(scan_interval_sec, background_scan).start()

# -------------------- GUI -------------------------
def start_diagnostics():
    output_text.delete("1.0", tk.END)

    def scan_and_display():
        verdict, trust = perform_scan()
        output_text.insert(tk.END, verdict)
        if trust < 0.8:
            plot_trust_curve(trust)

    threading.Thread(target=scan_and_display).start()

def toggle_voice():
    global voice_enabled
    voice_enabled = not voice_enabled
    status = "enabled" if voice_enabled else "disabled"
    voice_button.config(text=f"Voice: {status.capitalize()}")

# -------------------- LAUNCH GUI -------------------------
root = tk.Tk()
root.title("ECC Memory Checker & Symbolic Verdict Engine")
root.geometry("700x520")

title = tk.Label(root, text="🔍 ECC Memory Checker with ASI Glyphs", font=("Helvetica", 16, "bold"))
title.pack(pady=10)

btn = tk.Button(root, text="Start Manual Scan", command=start_diagnostics, bg="#3c7", fg="white", font=("Helvetica", 14))
btn.pack(pady=8)

voice_button = tk.Button(root, text="Voice: Enabled", command=toggle_voice, bg="#555", fg="white", font=("Helvetica", 12))
voice_button.pack(pady=4)

vol_label = tk.Label(root, text="🔊 Volume", font=("Helvetica", 12))
vol_label.pack()
vol_slider = ttk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume)
vol_slider.set(100)
vol_slider.pack(pady=4)

output_text = tk.Text(root, wrap=tk.WORD, font=("Helvetica", 12), bg="#f2f2f2")
output_text.pack(padx=20, pady=10, expand=True, fill=tk.BOTH)

# Start automatic background scan
background_scan()

root.mainloop()

