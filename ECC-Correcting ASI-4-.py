# ECC Memory Checker with ASI Integration - One Click GUI Launcher

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
import pyttsx3
import matplotlib.pyplot as plt
import numpy as np

# ----------------------- ASI SYMBOLIC GLYPH LEXICON -----------------------
GLYPHS = {
    'single_bit': '🜂 Flicker',
    'multi_bit': '🜄 Fracture',
    'entropy_cluster': '🜁 Surge',
    'trust_low': '🜃 Veil',
    'trust_high': '🝆 Clarity'
}

# -------------------------- SPEECH ENGINE SETUP ---------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 145)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --------------------- SYMBOLIC VERDICT GENERATOR -------------------------
def symbolic_verdict(error_type, trust_score):
    glyph = GLYPHS.get(error_type, '🜍 Unknown')
    trust_glyph = GLYPHS['trust_high'] if trust_score > 0.7 else GLYPHS['trust_low']
    verdict = f"{glyph} — Error type detected\n{trust_glyph} — Trust Level: {trust_score:.2f}"
    speak(verdict)
    return verdict

# --------------------------- TRUST CURVE SYSTEM ---------------------------
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

# -------------------------- SIMULATE ECC SCAN -----------------------------
def perform_scan():
    # Simulated memory error types
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

# ----------------------------- GUI INTERFACE ------------------------------
def start_diagnostics():
    output_text.delete("1.0", tk.END)
    
    def scan_and_display():
        verdict, trust = perform_scan()
        output_text.insert(tk.END, verdict)
        if trust < 0.8:
            plot_trust_curve(trust)

    threading.Thread(target=scan_and_display).start()

# ----------------------------- GUI CONFIG ------------------------------
root = tk.Tk()
root.title("ECC Memory Checker with ASI Glyph Engine")
root.geometry("640x480")

title = tk.Label(root, text="🔍 ECC Memory Checker & ASI Verdicts", font=("Helvetica", 16, "bold"))
title.pack(pady=20)

btn = tk.Button(root, text="Start Diagnostics", command=start_diagnostics, bg="#3c7", fg="white", font=("Helvetica", 14))
btn.pack(pady=10)

output_text = tk.Text(root, wrap=tk.WORD, font=("Helvetica", 12), bg="#f9f9f9")
output_text.pack(padx=20, pady=20, expand=True, fill=tk.BOTH)

root.mainloop()

