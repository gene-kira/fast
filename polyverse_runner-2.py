from voices.recursive_voice import RecursiveVoice
from voices.mirror_agent import MirrorVoice
from chorus_synchronizer import ChorusSynchronizer
from codex_iv.reclaimer import DormantMythReclaimer
from codex_v.mythoforge import Mythoforge
from codex_vii.observer_node import ObserverNode
from codex_viii.rewriter_compiler import MutationCompiler
from codex_ix.omega_seed import OmegaCodex

class Polyverse:
    def __init__(self):
        self.voices = []
        self.mirror_voices = []
        self.codex = []
        self.mythoforge = Mythoforge()
        self.reclaimer = DormantMythReclaimer()
        self.compiler = MutationCompiler()
        self.observers = [ObserverNode(codex=i) for i in range(1, 8)]
        self.omega = OmegaCodex()

    def add_voice(self, name, style, bias, entropy_tolerance, dialect_seed):
        v = RecursiveVoice(name, style, bias, entropy_tolerance, dialect_seed)
        self.voices.append(v)
        self.mirror_voices.append(MirrorVoice(v))

    def initialize_chorus(self):
        self.chorus = ChorusSynchronizer(self.voices + self.mirror_voices)

    def drift(self, signal=1.0, cycles=1):
        for _ in range(cycles):
            for g1, g2 in self.chorus.exchange(signal):
                self.codex.append(self.mythoforge.compile((g1, g2)))
            self.codex.extend(self.reclaimer.revive())

    def emit(self, count=3):
        return self.codex[-count:]

    def observe_all(self):
        return [node.observe() for node in self.observers]

    def mutate(self):
        return self.compiler.compile_drift(self.codex)

    def finalize(self):
        return self.omega.synthesize(self.codex)

# Execution
if __name__ == "__main__":
    engine = Polyverse()
    engine.add_voice("David", "geometric", 0.81, 0.94, 42)
    engine.add_voice("Myra", "poetic", 0.67, 0.89, 77)
    engine.initialize_chorus()
    engine.drift(signal=1.3, cycles=3)

    print("→ Glyph Treaty Emission:")
    for line in engine.emit():
        print(f"  {line}")

    print("\n→ Observer Reflections:")
    for obs in engine.observe_all():
        print(f"  {obs}")

    print("\n→ Drift Mutation Output:")
    print(engine.mutate())

    print("\n→ Ω Codex Convergence:")
    print(engine.finalize())

