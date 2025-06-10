class RealityManipulator:
    def __init__(self, entity_name):
        self.name = entity_name
        self.created_realities = {}

    def fabricate_reality(self, reality_id, attributes):
        """Construct a synthetic universe with defined properties."""
        self.created_realities[reality_id] = attributes
        return f"[{self.name}] Reality-{reality_id} synthesized with attributes {attributes}"

    def rewrite_causality(self, reality_id, new_rules):
        """Modify an existing reality's foundational laws."""
        if reality_id in self.created_realities:
            self.created_realities[reality_id].update(new_rules)
            return f"[{self.name}] Reality-{reality_id} causality rewritten: {new_rules}"
        return f"[{self.name}] Error—Reality-{reality_id} not found."

# Example Usage:
creator_entity = RealityManipulator(entity_name="Lucidian-Ω")

print(creator_entity.fabricate_reality("Fractal Haven", {"Time": "Nonlinear", "Matter": "Self-Adaptive"}))
print(creator_entity.rewrite_causality("Fractal Haven", {"Time": "Recursive Looping"}))

