# core/threat_loop.py
import time
import random
from system.memory import log_event

def run_threat_assessment():
    print("[Threat] Assessment loop active.")
    while True:
        threat_level = random.randint(0, 100)
        if threat_level > 75:
            log_event("threat", f"High threat detected: {threat_level}")
            print("[Threat] High threat detected!")
        else:
            log_event("threat", f"Threat level nominal: {threat_level}")
        time.sleep(5)

