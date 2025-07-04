# === PART 1 — AUTOLOADER CORE SETUP ===
import sys
import subprocess
import importlib

# List of required packages
required_packages = {
    "uuid", "time", "random", "hashlib", "threading", "socket",
    "tkinter", "pyttsx3", "flask", "cryptography", "psutil",
    "platform", "yaml", "joblib", "sklearn", "onnxruntime",
    "matplotlib", "requests", "numpy"
}

# Fallback mapping if module name != pip name
pip_rename = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "onnxruntime": "onnxruntime",
}

def auto_install():
    for module in required_packages:
        try:
            importlib.import_module(module)
        except ImportError:
            pip_name = pip_rename.get(module, module)
            print(f"📦 Installing missing package: {pip_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

auto_install()

# === PART 2 — BASE IMPORT LAYER ===
import uuid, time, random, hashlib, threading, socket
import platform, sys, os
import tkinter as tk
from tkinter import ttk, scrolledtext
import pyttsx3
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet
import psutil
import requests
import yaml
import joblib
import onnxruntime as rt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

# === PART 3 — DIAGNOSTIC + BOOT CHECK ===

def diagnostic_report():
    print("\n🛡️ AUTORUN SYSTEM CHECK")
    for mod in sorted(required_packages):
        try:
            __import__(mod)
            print(f"✅ {mod} ready.")
        except ImportError:
            print(f"❌ {mod} missing.")

    print("\n🔐 Cryptography backend: ", Fernet.generate_key().decode()[:10], "...ready")

diagnostic_report()

# === PART 4 — ENTRY POINT BINDING ===

def launch_arkforge():
    from core.vault import Vault
    from core.kernel import ASIKernel
    from core.glyph import GlyphEngine
    from core.orchestrator import SigilOrchestrator
    from core.trust import ThreatClassifier
    from ui.hud import HUD
    from util.speech import speak
    from net.replicate import vault_listener, broadcast_presence
    from glyph_loader import load_glyphbook, bind_glyphs
    from reflex import reflex_filter

    vault = Vault()
    asi = ASIKernel(vault)
    classifier = ThreatClassifier()
    glyphs = GlyphEngine()
    orchestrator = SigilOrchestrator(asi)

    book = load_glyphbook("glyphbook.yaml")
    bind_glyphs(glyphs, orchestrator, book)
    asi.install_filter(reflex_filter(asi, classifier))

    threading.Thread(target=vault_listener, args=(asi,), daemon=True).start()
    threading.Thread(target=broadcast_presence, daemon=True).start()

    hud = HUD(asi, orchestrator)
    hud.run()

if __name__ == "__main__":
    launch_arkforge()

