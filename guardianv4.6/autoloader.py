import subprocess
import sys
import platform
import os

REQUIRED_PACKAGES = ["cryptography", "PyQt5"]

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[AUTOLOADER] Installed: {package}")
    except subprocess.CalledProcessError:
        print(f"[AUTOLOADER] Failed to install: {package}")

def autoload():
    print("🔍 Checking system environment...")
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            print(f"[AUTOLOADER] Found: {package}")
        except ImportError:
            print(f"[AUTOLOADER] Missing: {package} — installing...")
            install_package(package)

if __name__ == "__main__":
    autoload()

