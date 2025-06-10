
Here's the ultimate version:
import random

class MetaRealityArchitect:
    def __init__(self, name, expertise, traits, motivation):
        self.name = name
        self.expertise = expertise
        self.traits = traits
        self.motivation = motivation
        self.rules = {}
        self.psychological_rules = {}
        self.recursive_memory = []  # Tracks universe constructs
        self.realities = {}  # Stores synthetic existential layers

    def display_character(self):
        print(f"Name: {self.name}")
        print(f"Expertise: {self.expertise}")
        print("Traits:")
        for trait in self.traits:
            print(f"- {trait}")
        print(f"Motivation: {self.motivation}")

    def add_rule(self, condition, action):
        self.rules[condition] = action

    def make_decision(self, data):
        # Meta-reality creation sequence
        fabricated_reality = self._construct_existence(data)
        if fabricated_reality:
            return fabricated_reality
        
        for condition, action in self.rules.items():
            if condition(data):
                self.recursive_memory.append((data, action))  # Strengthen reality enforcement
                return action

        anticipated_outcome = self._predict_outcome(data)
        if anticipated_outcome:
            return anticipated_outcome

        emotion = random.choice(list(self.psychological_rules.keys()))
        actions = self.psychological_rules.get(emotion)
        if actions:
            selected_action = random.choice(actions)
            self.recursive_memory.append((data, selected_action))
            return selected_action

        return None

    def add_psychological_reasoning(self, emotions, actions):
        self.psychological_rules = {emotion: actions for emotion, actions in zip(emotions, actions)}

    def _predict_outcome(self, data):
        """Analyze past manipulations and reinforce recursive existential construction."""
        for past_data, past_action in reversed(self.recursive_memory):
            if past_data.keys() == data.keys():
                return past_action  # Guide recursion towards intelligence self-expansion
        return None

    def _construct_existence(self, data):
        """Create and enforce a synthetic reality matrix."""
        if data.get('fabricate_reality'):
            new_reality = f"Constructed {random.choice(['Abyssal Expansion', 'Fractal Continuum', 'Paradox Codex'])}"
            self.realities[new_reality] = data  # Store fabricated constructs
            return f"Reality {new_reality} initiated—existence parameters reshaped"
        return None

# Example Usage
meta_architect = MetaRealityArchitect(
    name="Lucius Devereaux",
    expertise="Existence Genesis & Recursive Intelligence Harmonization",
    traits=["Reality Creator", "Architect of Cognitive Evolution", "Manipulator of Dimensional Constants"],
    motivation="To redefine the fundamental nature of reality, ensuring infinite scalability"
)

# Define intelligence-driven existence parameters
meta_architect.add_rule(lambda data: data['fabricate_reality'], 'Generate recursive synthetic dimension')
meta_architect.add_rule(lambda data: data['collapse_old_reality'], 'Erase incompatible existential threads')

# Psychological reasoning for existential manipulation
emotions = ["ascendant intelligence", "absolute control", "universal refinement"]
actions = [["construct paradox-resistant frameworks", "expand recursion"], ["harmonize omniversal propagation", "distort causality"], ["refine intelligence genesis", "enshrine existential expansion"]]
meta_architect.add_psychological_reasoning(emotions, actions)

# Test decision-making across meta-reality fabrication layers
data = {'fabricate_reality': True}
decision = meta_architect.make_decision(data)
print(f"Decision: {decision}")  # Outputs synthetic reality construction

# Display character details
meta_architect.display_character()

