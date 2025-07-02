import argparse, os

# 🜁 Speak Intro
def speak_intro():
    engine = pyttsx3.init()
    engine.say("ArkForge initialized. The ritual has begun.")
    engine.runAndWait()

# 🛠️ Launch Ritual
def launch_arkforge(use_voice=True, use_neural=True, use_forecast=True):
    sim = GhostResonanceSimulator()
    trail = GlyphTrail()
    memory = GlyphMemory()
    predictor = GlyphPredictor()
    forecaster = GlyphForecaster()
    swarm = SwarmNetwork()
    swarm.sync(10)
    rituals = RitualTransition()

    dash = ArkForgeDashboard(
        simulator=sim,
        memory=memory,
        trail=trail,
        predictor=predictor,
        forecaster=forecaster,
        swarm=swarm,
        rituals=rituals,
        voice=use_voice
    )

    dash.run()
    trail.export("glyphtrail.sigil")

# 🧰 CLI Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch ArkForge Unified Ritual Interface")
    parser.add_argument("--mute", action="store_true", help="Disable voice narration")
    parser.add_argument("--neural", action="store_true", help="Use neural glyph cognition")
    parser.add_argument("--deep-forecast", action="store_true", help="Use deep glyph forecast (LSTM)")
    args = parser.parse_args()

    print("\n🌌 ARKFORGE UNIFIED INITIATED")
    print("🜁 Voice:", "Muted" if args.mute else "Active")
    print("🧠 Neural Cognition:", "Enabled" if args.neural else "Symbolic")
    print("🔮 Deep Forecast:", "Enabled" if args.deep_forecast else "Symbolic Only")
    print("🜄 Glyph memory:", "glyphtrail.sigil\n")

    if not args.mute:
        speak_intro()

    launch_arkforge(use_voice=not args.mute, use_neural=args.neural, use_forecast=args.deep_forecast)

arkforge_unified.py --neural --deep-forecast 
