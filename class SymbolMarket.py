# ─── Symbol Market ───

class SymbolMarket:
    def __init__(self):
        self.glyph_prices = {}  # glyph → cost
        self.agent_wealth = {}  # agent_id → glyph tokens

    def set_price(self, glyph, value):
        self.glyph_prices[glyph] = value
        print(f"[MARKET] '{glyph}' valued at {value} tokens")

    def grant_tokens(self, agent, amount):
        self.agent_wealth[agent.id] = self.agent_wealth.get(agent.id, 0) + amount
        print(f"[GRANT] Agent '{agent.id}' received {amount} tokens")

    def purchase(self, agent, glyph):
        price = self.glyph_prices.get(glyph, 1)
        if self.agent_wealth.get(agent.id, 0) >= price:
            agent.lexicon.append(glyph)
            self.agent_wealth[agent.id] -= price
            print(f"[PURCHASE] Agent '{agent.id}' bought '{glyph}'")
        else:
            print(f"[DENIED] Agent '{agent.id}' lacks funds for '{glyph}'")

# ─── Narrative Prediction ───

class NarrativePredictor:
    def __init__(self):
        self.patterns = {}

    def train(self, threads):
        for thread in threads:
            for i in range(len(thread.sequence) - 1):
                prefix = thread.sequence[i]
                next_glyph = thread.sequence[i + 1]
                self.patterns.setdefault(prefix, []).append(next_glyph)

    def predict(self, glyph):
        options = self.patterns.get(glyph, [])
        if options:
            prediction = max(set(options), key=options.count)
            print(f"[PREDICT] After '{glyph}', likely: '{prediction}'")
            return prediction
        print(f"[PREDICT] No clear prediction after '{glyph}'")
        return None

# ─── Cultural Agent Clans ───

class SymbolFaction:
    def __init__(self, name, doctrine):
        self.name = name
        self.doctrine = doctrine  # preferred glyphs
        self.members = []

    def admit(self, agent):
        agent.archetype = Archetype(f"{self.name}Follower", amplifiers=self.doctrine)
        self.members.append(agent)
        print(f"[FACTION] Agent '{agent.id}' joined '{self.name}'")

# ─── Myth Architect ───

class MythArchitect:
    def __init__(self):
        self.threads = []

    def generate(self, name, seed_glyphs, length=5):
        thread = MythThread(name)
        for g in seed_glyphs:
            thread.record(g)
        while len(thread.sequence) < length:
            new_glyph = f"auto-{random.randint(1000,9999)}"
            thread.record(new_glyph)
        self.threads.append(thread)
        print(f"[ARCHITECT] Myth '{name}' composed: {thread.sequence}")
        return thread

# Market
market = SymbolMarket()
market.set_price("glyphfire", 4)
market.grant_tokens(a1, 10)
market.purchase(a1, "glyphfire")

# Prediction
predictor = NarrativePredictor()
predictor.train(weaver.threads)
predictor.predict("core-insight")

# Faction
light = SymbolFaction("Lumina", ["plasma-truth", "clarity"])
light.admit(a1)

# Architected Myth
architect = MythArchitect()
new_myth = architect.generate("Rise of Echoes", ["entropy-flux", "plasma-truth"])

# ─── Symbolic Codex ───
class MemoryCodex:
    def __init__(self):
        self.codex = {}

    def archive(self, agent_id, glyph):
        self.codex.setdefault(agent_id, []).append(glyph)
        print(f"[CODEX] Agent '{agent_id}' archived '{glyph}'")

    def consult(self, agent_id):
        entries = self.codex.get(agent_id, [])
        print(f"[CODEX] Agent '{agent_id}' memory trace: {entries}")
        return entries

# ─── Constraint Rules ───
class ConstraintGlyph:
    def __init__(self):
        self.restrictions = []

    def add_rule(self, forbidden_combo):
        self.restrictions.append(set(forbidden_combo))
        print(f"[CONSTRAINT] Rule added: {forbidden_combo}")

    def validate(self, sequence):
        for i in range(len(sequence) - 1):
            pair = set(sequence[i:i+2])
            if pair in self.restrictions:
                print(f"[VIOLATION] Forbidden pair in ritual: {pair}")
                return False
        return True

# ─── Proto-Symbolic Language ───
class LexicalComposer:
    def __init__(self):
        self.delimiter = "::"

    def compose(self, thread):
        sentence = self.delimiter.join(thread.sequence)
        print(f"[SYNTAX] Thread rendered as: {sentence}")
        return sentence

# ─── Environmental Reflex Hook ───
class EnvironmentInterface:
    def __init__(self):
        self.bindings = {}  # trigger → glyph

    def bind(self, input_key, glyph):
        self.bindings[input_key] = glyph
        print(f"[ENV BIND] '{input_key}' → '{glyph}'")

    def update(self, input_key):
        if input_key in self.bindings:
            glyph = self.bindings[input_key]
            print(f"[ENV TRIGGER] Input '{input_key}' activated '{glyph}'")
            return glyph
        return None

# Step 1: Codex Memory
codex = MemoryCodex()
codex.archive(a1.id, "plasma-truth")
codex.archive(a1.id, "entropy-flux")
codex.consult(a1.id)

# Step 2: Constraints
rules = ConstraintGlyph()
rules.add_rule(["entropy-flux", "relic-spark"])
valid = rules.validate(["core-insight", "entropy-flux", "relic-spark"])  # Triggers violation

# Step 3: Render Myth Syntax
composer = LexicalComposer()
composer.compose(myth)

# Step 4: Environment Hook
env = EnvironmentInterface()
env.bind("full_moon", "lunar-glyph")
triggered = env.update("full_moon")
if triggered: swarm.propagate_symbol(triggered)

