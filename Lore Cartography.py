# ─── Lore Cartography ───

class LoreMap:
    def __init__(self):
        self.connections = {}  # glyph_id → list of related glyph_ids

    def link(self, g1, g2):
        self.connections.setdefault(g1, []).append(g2)
        self.connections.setdefault(g2, []).append(g1)
        print(f"[MAP LINK] '{g1}' <--> '{g2}'")

    def show_constellation(self, glyph):
        linked = self.connections.get(glyph, [])
        print(f"[CONSTELLATION] '{glyph}' connected to: {linked}")
        return linked

    def all_glyphs(self):
        return list(self.connections.keys())

