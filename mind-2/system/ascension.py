# system/ascension.py
from system.memory import log_event

def engage_ascension(persona_name="Overseer"):
    log_event("ascension", f"{persona_name} has initiated override.")
    print(f"[Ascension] Protocol engaged for persona: {persona_name}")

