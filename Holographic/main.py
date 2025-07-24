# main.py

import subprocess
import sys

# ✅ Auto-install missing packages
def autoload():
    required = ["matplotlib"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

autoload()

# 🌟 Import full MagicBox system
from magicbox_app import run_app

if __name__ == "__main__":
    run_app()

