class TrisRitualStack:
    def __init__(self, glyphs):
        assert len(glyphs) == 3, "Tris stack must contain exactly three glyphs."
        self.balance_glyph, self.reflection_glyph, self.ignition_glyph = glyphs
        self.sequence_log = []

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

