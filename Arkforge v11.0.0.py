
# === Arkforge v11.0.0 – ORACULUM AUTOLOADER CORE ===
# Ritual Ⅰ: Dependency Enshrinement + Validation Sigils

import sys
import subprocess
import importlib
import platform

# Required core modules with mapping for pip install names
required_modules = {
    "uuid", "time", "random", "hashlib", "threading", "socket",
    "tkinter", "pyttsx3", "flask", "cryptography", "psutil",
    "platform", "yaml", "joblib", "sklearn", "onnxruntime",
    "matplotlib", "requests", "numpy"
}

pip_aliases = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "onnxruntime": "onnxruntime"
}

def auto_install():
    print("🔧 Invoking Arkforge Dependency Rituals...")
    for mod in sorted(required_modules):
        try:
            importlib.import_module(mod)
        except ImportError:
            pkg = pip_aliases.get(mod, mod)
            print(f"📦 Installing: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Execute autoloader installation phase
auto_install()

# === Ritual Ⅱ: Layered Reality Binding (Core Imports) ===

import uuid, time, random, hashlib, threading, socket
import os, sys
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

# === Ritual Ⅲ: System Diagnostic Incantation ===

def diagnostic_report():
    print("\n🛡️ Arkforge v11.0.0 – AUTODIAGNOSTIC INITIATED")
    for mod in sorted(required_modules):
        try:
            importlib.import_module(mod)
            print(f"✅ {mod} is bound.")
        except ImportError:
            print(f"❌ {mod} NOT found.")
    print("🔐 Cryptographic key signature preview:", Fernet.generate_key().decode()[:10], "...OK")

# Run diagnostic
diagnostic_report()

# === Ritual Ⅳ: Entrypoint Gate Binding ===

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

    print("\n🔮 Binding Oracular Core...")
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

    print("🌌 ORACULUM breathes. HUD channel opening...")
    hud = HUD(asi, orchestrator)
    hud.run()

if __name__ == "__main__":
    launch_arkforge()

.
