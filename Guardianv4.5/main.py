
import subprocess
import sys

# List of required packages
REQUIRED_PACKAGES = [
    "cryptography",
    "PyQt5"
]

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[AUTOLOADER] Installed: {package}")
    except subprocess.CalledProcessError:
        print(f"[AUTOLOADER] Failed to install: {package}")

def autoload():
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            print(f"[AUTOLOADER] Found: {package}")
        except ImportError:
            print(f"[AUTOLOADER] Missing: {package} — installing...")
            install_package(package)

if __name__ == "__main__":
    autoload()






import sys
from PyQt5.QtWidgets import QApplication
from guardian_gui import GuardianGUI

def main():
    app = QApplication(sys.argv)
    gui = GuardianGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

