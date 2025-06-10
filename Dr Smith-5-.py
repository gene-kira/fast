

### Character Class with Decision-Making Capabilities

```python
class Character:
    def __init__(self, name, field_of_expertise, traits, motivation):
        self.name = name
        self.field_of_expertise = field_of_expertise
        self.traits = traits
        self.motivation = motivation
        self.rules = {}
        self.psychological_rules = {}

    def display_character(self):
        print(f"Name: {self.name}")
        print(f"Field of Expertise: {self.field_of_expertise}")
        print("Traits:")
        for trait in self.traits:
            print(f"- {trait}")
        print(f"Motivation: {self.motivation}")

    def add_rule(self, condition, action):
        self.rules[condition] = action

    def make_decision(self, data):
        # Logical decision-making
        for condition, action in self.rules.items():
            if condition(data):
                return action
        
        # Psychological reasoning
        emotion = choice(list(self.psychological_rules.keys()))
        actions = self.psychological_rules.get(emotion)
        if actions:
            return choice(actions)
        
        return None

    def add_psychological_reasoning(self, emotions, actions):
        self.psychological_rules = {emotion: actions for emotion, actions in zip(emotions, actions)}

# Example usage
from random import choice

hannibal_like_character = Character(
    name="Dr. Smith",
    field_of_expertise="Psychiatry and Culinary Arts",
    traits=["Highly intelligent", "Charismatic", "Manipulative"],
    motivation="A personal philosophy that justifies extreme actions"
)

# Define some rules
hannibal_like_character.add_rule(lambda data: data['threat_level'] > 5, 'neutralize threat')
hannibal_like_character.add_rule(lambda data: data['resource_needed'], 'acquire resources')

# Add psychological reasoning
emotions = ["calm", "angry", "curious"]
actions = [["observe", "analyze"], ["confront", "retreat"], ["investigate", "experiment"]]
hannibal_like_character.add_psychological_reasoning(emotions, actions)

# Make a decision
data = {'threat_level': 7, 'resource_needed': False}
decision = hannibal_like_character.make_decision(data)
print(f"Decision: {decision}")  # Output: neutralize threat

# Display character details
hannibal_like_character.display_character()
```

