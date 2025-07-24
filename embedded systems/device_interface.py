import sys
import os
import platform

def auto_detect_com():
    system = platform.system()
    if system == "Windows":
        from serial.tools import list_ports
        for port in list_ports.comports():
            if "USB" in port.description.upper() or "UART" in port.description.upper():
                return port.device

    elif system == "Linux":
        # Look for ttyUSBx or ttyACMx
        for dev in os.listdir("/dev"):
            if dev.startswith("ttyUSB") or dev.startswith("ttyACM"):
                return f"/dev/{dev}"

    elif system == "Darwin":  # macOS
        for dev in os.listdir("/dev"):
            if dev.startswith("cu.usbserial") or dev.startswith("tty.usbmodem"):
                return f"/dev/{dev}"

    return None

