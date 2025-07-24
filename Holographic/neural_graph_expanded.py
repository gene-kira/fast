# neural_graph_expanded.py

import random

class NeuralGraph:
    def __init__(self):
        self.zones = {
            "pulse.glyph": [],
            "deep.glyph": [],
            "reflect.glyph": []
        }
        self.memory_decay_rate = 0.98

    def link_probe(self, probe):
        glyph = probe.data_type
        confidence = probe.signal * 0.9
        match_score = confidence + random.uniform(-0.05, 0.05)

        node_data = {
            "confidence": round(match_score, 2),
            "status": "linked",
            "frame": glyph,
            "distortion": random.choice(["none", "noise", "overlay"])
        }

        self.zones[glyph].append(node_data)
        return node_data

