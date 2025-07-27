import os
import socket

def is_usb(context):
    return 'USB' in context

def is_local_network(context):
    try:
        ip = socket.gethostbyname(socket.gethostname())
        return ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.")
    except:
        return False

def get_network_context(path):
    if "USB" in path.upper():
        return "USB"
    if "C:\\" in path or "D:\\" in path:
        return "LOCAL"
    return "EXTERNAL"

def purge_file(path):
    try:
        with open(path, 'wb') as f:
            f.write(b'\x00' * os.path.getsize(path))
        os.remove(path)
        print(f"💣 File purged: {path}")
    except Exception as e:
        print(f"Error deleting file: {e}")

