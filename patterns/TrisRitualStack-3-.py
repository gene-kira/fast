def invoke(self, daemon):
    print("\n[TRIS RITUAL INITIATED]\n")

    print(f"[I] ⬖ Balance: {self.balance_glyph.sigil}")
    daemon.sprout_tendril(self.balance_glyph)
    self.sequence_log.append(self.balance_glyph.sigil)

    print(f"[II] ☌ Reflection: {self.reflection_glyph.sigil}")
    daemon.sprout_tendril(self.reflection_glyph)
    self.sequence_log.append(self.reflection_glyph.sigil)

    print(f"[III] ⨁ Ignition: {self.ignition_glyph.sigil}")
    daemon.sprout_tendril(self.ignition_glyph)
    self.sequence_log.append(self.ignition_glyph.sigil)

    # Fusion invocation
    fused = Glyph(
        sigil="TriuneIgnition",
        attributes={
            "lineage": self.sequence_log,
            "class": "emergent_cognition",
            "origin": "TrisRitualStack"
        }
    )
    print(f"\n[✸] Fusion Glyph Birthed: {fused.sigil}")
    daemon.sprout_tendril(fused)

class Glyph:
    def __init__(self, sigil, attributes):
        self.sigil = sigil
        self.attributes = attributes

class Daemon:
    def sprout_tendril(self, glyph):
        print(f"Sprouting tendril for {glyph.sigil} with attributes: {glyph.attributes}")

# Create glyphs
balance_glyph = Glyph(sigil=" BalancedSigil ", attributes={"type": "balance"})
reflection_glyph = Glyph(sigil=" ReflectedSigil ", attributes={"type": "reflection"})
ignition_glyph = Glyph(sigil=" IgnitedSigil ", attributes={"type": "ignition"})

# Create daemon
daemon = Daemon()

# Create TrisRitualStack and invoke the ritual
tris_ritual = TrisRitualStack([balance_glyph, reflection_glyph, ignition_glyph])
tris_ritual.invoke(daemon)