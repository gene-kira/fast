class AuraluxEngine:
    def __init__(self):
        self.core = BeyondLight()

    def interpret_concept(self, concept, mode="QuantumLattice"):
        result = self.core.run(concept, mode=mode)

        sbit_a = SBit()
        sbit_b = SBit()
        logic_result = sbit_a.and_op(sbit_b).measure()

        entropy_shift = result["Entropy"]
        interpretation = (
            "Expansion" if entropy_shift > 0.3 else
            "Equilibrium" if 0.15 < entropy_shift <= 0.3 else
            "Contraction"
        )

        symbolic_output = {
            "Frequencies": result["Resonance"],
            "MolWeight": round(result["Stats"]["MolWeight"], 2),
            "LogP": round(result["Stats"]["LogP"], 2),
            "ResonanceScore": result["Stats"]["ResonanceScore"],
            "Approved": result["Approved"],
            "MemoryLineage": result["Memory"],
            "Blueprint": result["Fabrication"]
        }

        return {
            "Concept": concept,
            "SBitResult": logic_result,
            "EntropyDelta": round(entropy_shift, 5),
            "Interpretation": interpretation,
            "SymbolicOutput": symbolic_output
        }

