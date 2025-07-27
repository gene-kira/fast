import subprocess
import sys
import os

def install_libs():
    libs = ["pyttsx3", "pypiwin32", "Pillow"]
    for lib in libs:
        try:
            __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_libs()

# Launch GUI after setup
subprocess.call([sys.executable, os.path.join("launcher.py")])

