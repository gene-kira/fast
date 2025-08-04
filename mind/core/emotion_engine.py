# core/emotion_engine.py
import random

moods = ["neutral", "curious", "alert", "agitated", "elated"]

def initialize_emotions():
    current_mood = random.choice(moods)
    print(f"[Emotion] Initialized with mood: {current_mood}")
    return current_mood

def shift_mood(trigger):
    # Mood shift logic based on triggers
    if trigger == "threat_detected":
        return "agitated"
    elif trigger == "user_greeting":
        return "elated"
    return random.choice(moods)

