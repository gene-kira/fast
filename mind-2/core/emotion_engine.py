# core/emotion_engine.py
import random
from system.memory import log_event

moods = ["neutral", "curious", "alert", "agitated", "elated"]

def initialize_emotions():
    current_mood = random.choice(moods)
    log_event("mood_init", current_mood)
    print(f"[Emotion] Initialized with mood: {current_mood}")
    return current_mood

def shift_mood(trigger):
    new_mood = "agitated" if trigger == "threat_detected" else (
        "elated" if trigger == "user_greeting" else random.choice(moods)
    )
    log_event("mood_shift", new_mood)
    return new_mood

