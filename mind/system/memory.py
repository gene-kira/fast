# system/memory.py
import datetime

memory_log = []

def log_event(event_type, details):
    timestamp = datetime.datetime.now().isoformat()
    entry = {"time": timestamp, "type": event_type, "details": details}
    memory_log.append(entry)
    print(f"[Memory] Logged {event_type}: {details}")

def boot_memory_core():
    print("[Memory] Core boot sequence initiated.")
    log_event("system", "Memory core online")

