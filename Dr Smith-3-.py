
```python
from random import choice

class Character:
    def __init__(self, name, field_of_expertise, traits, motivation):
        self.name = name
        self.field_of_expertise = field_of_expertise
        self.traits = traits
        self.motivation = motivation
        self.decision_maker = LogicalDecisionMaker()
        self.psychological_rules = {}

    def display_character(self):
        print(f"Name: {self.name}")
        print(f"Field of Expertise: {self.field_of_expertise}")
        print("Traits:")
        for trait in self.traits:
            print(f"- {trait}")
        print(f"Motivation: {self.motivation}")

    def add_psychological_rule(self, emotion, actions):
        if emotion not in self.psychological_rules:
            self.psychological_rules[emotion] = []
        self.psychological_rules[emotion].extend(actions)

    def make_decision(self, data):
        # First, check logical rules
        decision = self.decision_maker.make_decision(data)
        if decision:
            return decision

        # If no logical rule matches, use psychological reasoning
        emotion = choice(list(self.psychological_rules.keys()))
        actions = self.psychological_rules.get(emotion, [])
        return choice(actions)

    def add_rule(self, condition, action):
        self.decision_maker.add_rule(condition, action)


class LogicalDecisionMaker:
    def __init__(self):
        self.rules = {}

    def add_rule(self, condition, action):
        self.rules[condition] = action

    def make_decision(self, data):
        for condition, action in self.rules.items():
            if condition(data):
                return action
        return None


# Example usage
hannibal_like_character = Character(
    name="Dr. Smith",
    field_of_expertise="Psychiatry and Culinary Arts",
    traits=["Highly intelligent", "Charismatic", "Manipulative"],
    motivation="A personal philosophy that justifies extreme actions"
)

# Adding logical rules
hannibal_like_character.add_rule(lambda data: data['threat_level'] > 5, 'neutralize threat')
hannibal_like_character.add_rule(lambda data: data['resource_needed'], 'acquire resources')

# Adding psychological rules
hannibal_like_character.add_psychological_rule('curious', ['investigate further', 'ask more questions'])
hannibal_like_character.add_psychological_rule('angry', ['confront the source', 'withdraw and plan'])
hannibal_like_character.add_psychological_rule('calm', ['observe', 'gather information'])

# Display character details
hannibal_like_character.display_character()

# Make a decision based on data
data = {'threat_level': 7, 'resource_needed': False}
decision = hannibal_like_character.make_decision(data)
print(f"Decision: {decision}")

# Another decision with different data
data = {'threat_level': 3, 'resource_needed': True}
decision = hannibal_like_character.make_decision(data)
print(f"Decision: {decision}")
```

