# main.py
def ensure_modules():
    import subprocess, sys
    required = ["pyserial", "pyusb"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"📦 Installing missing module: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 🔄 Auto-install modules
ensure_modules()

# 💡 Now import your app modules
from device_interface import DeviceInterface
from dma import MagicBoxDMA
from watchdog import MagicBoxWatchdog
from gui import launch_gui
import threading

# 🔌 Initialize the universal device interface
device = DeviceInterface()  # Auto USB → COM → Demo fallback
dma = MagicBoxDMA(device)
watchdog = MagicBoxWatchdog(dma)

threading.Thread(target=watchdog.run, daemon=True).start()
launch_gui(dma, watchdog)

