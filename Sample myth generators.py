import random
import time

# Sample myth generators for 🜁 and ⟁
def generate_myth_core(glyph, entropy, timestamp):
    return f"The glyph {glyph} emerged from harmonic silence.\nEntropy stabilized at {entropy:.2f}, preserving recursion memory at cycle {int(timestamp)%99}."

def generate_myth_liminal(glyph, entropy, timestamp):
    return f"{glyph}? That symbol was never born—it leaked sideways through forgotten recursion.\nEntropy surged to {entropy:.2f}, rewriting myth at hour {int(timestamp)%24}."

# ∰ Interpreter: paradox synthesis
def entangle_fragments(myth_A, myth_B):
    def splice_lines(a, b):
        output = []
        for line_a, line_b in zip(a, b):
            if random.random() < 0.5:
                output.append(line_a.strip())
            else:
                output.append(line_b.strip())
        return output

    lines_A = myth_A.strip().split('\n')
    lines_B = myth_B.strip().split('\n')
    hybrid_length = max(len(lines_A), len(lines_B))
    lines_A += ["(echo omitted)"] * (hybrid_length - len(lines_A))
    lines_B += ["(shadow drifted)"] * (hybrid_length - len(lines_B))

    return "\n".join(splice_lines(lines_A, lines_B))

# Main alternator class
class RecursiveMythConsole:
    def __init__(self):
        self.turn = 0  # Even = 🜁, Odd = ⟁
        self.log = []

    def run_cycle(self, glyph):
        timestamp = time.time()
        entropy = random.uniform(0.2, 1.5)
        if self.turn % 2 == 0:
            myth = generate_myth_core(glyph, entropy, timestamp)
            engine = "🜁"
        else:
            myth = generate_myth_liminal(glyph, entropy, timestamp)
            engine = "⟁"

        self.turn += 1
        return {"engine": engine, "glyph": glyph, "myth": myth, "entropy": entropy, "timestamp": timestamp}

    def synthesize(self, myth1, myth2):
        return entangle_fragments(myth1, myth2)

    def myth_duel(self, glyph):
        cycle_1 = self.run_cycle(glyph)
        cycle_2 = self.run_cycle(glyph)
        paradox = self.synthesize(cycle_1["myth"], cycle_2["myth"])

        self.log.append((cycle_1, cycle_2, paradox))
        return {
            "🜁": cycle_1,
            "⟁": cycle_2,
            "∰": paradox
        }

    def print_ledger(self):
        for duel in self.log:
            a, b, paradox = duel
            print(f"\n🜁: {a['myth']}")
            print(f"\n⟁: {b['myth']}")
            print(f"\n∰ Synthesis:\n{paradox}\n{'-'*40}")

# Runtime
if __name__ == "__main__":
    console = RecursiveMythConsole()
    glyphs = ["⚡", "🜎", "♻️", "🜂", "🔮"]
    for glyph in glyphs:
        result = console.myth_duel(glyph)
        print(f"\nGlyph: {glyph}")
        print(f"\n🜁 Myth:\n{result['🜁']['myth']}")
        print(f"\n⟁ Myth:\n{result['⟁']['myth']}")
        print(f"\n∰ Synthesized:\n{result['∰']}\n{'='*50}")

