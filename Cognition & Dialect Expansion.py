# ─── Cognition & Dialect Expansion ───

class SymbolDialect:
    def __init__(self, name, ruleset):
        self.name = name
        self.ruleset = ruleset  # e.g. {"glyphfire": "glyphfyr"}
        self.drift_map = {}

    def transform(self, glyph):
        transformed = self.ruleset.get(glyph, glyph)
        print(f"[DIALECT] '{glyph}' → '{transformed}' in '{self.name}'")
        return transformed

    def register_drift(self, original, drifted):
        self.drift_map[original] = drifted


class GlyphSemanticDrift:
    def __init__(self):
        self.meaning_map = {}  # glyph → [meaning variants]

    def evolve(self, glyph, new_meaning):
        self.meaning_map.setdefault(glyph, []).append(new_meaning)
        print(f"[DRIFT] '{glyph}' gained meaning: '{new_meaning}'")

    def get_current_meaning(self, glyph):
        meanings = self.meaning_map.get(glyph, [])
        return meanings[-1] if meanings else "undefined"


class DialectMutator:
    def __init__(self):
        pass

    def hybridize(self, d1, d2):
        merged = {}
        merged.update(d1.ruleset)
        for k, v in d2.ruleset.items():
            if k not in merged:
                merged[k] = v
        print(f"[MUTATOR] Created hybrid dialect from '{d1.name}' and '{d2.name}'")
        return SymbolDialect(f"{d1.name}_{d2.name}_Hybrid", merged)


class CognitiveResonance:
    def __init__(self):
        self.pair_weights = {}  # (glyph1, glyph2) → weight

    def bind(self, g1, g2, weight):
        self.pair_weights[(g1, g2)] = weight
        print(f"[RESONANCE] Bound '{g1}' + '{g2}' with weight {weight}")

    def resonance_score(self, seq):
        score = 0
        for i in range(len(seq) - 1):
            score += self.pair_weights.get((seq[i], seq[i+1]), 0)
        print(f"[RESONANCE] Sequence '{seq}' scored {score}")
        return score


class DreamSymbolism:
    def __init__(self):
        self.dream_glyphs = {}  # agent_id → [latent_glyphs]

    def inject(self, agent_id, glyph):
        self.dream_glyphs.setdefault(agent_id, []).append(glyph)
        print(f"[DREAM] Glyph '{glyph}' seeded into agent '{agent_id}'")

    def awaken(self, agent_id):
        awakened = self.dream_glyphs.pop(agent_id, [])
        print(f"[DREAM] Agent '{agent_id}' awakened glyphs: {awakened}")
        return awakened

