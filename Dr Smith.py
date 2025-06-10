

Here's an enhanced version of the `Character` class:

```python
from random import choice

class Character:
    def __init__(self, name, field_of_expertise, traits, motivation):
        self.name = name
        self.field_of_expertise = field_of_expertise
        self.traits = traits
        self.motivation = motivation
        self.decision_maker = LogicalDecisionMaker()
        self.psychological_reasoning = PsychologicalReasoningEngine()

    def display_character(self):
        print(f"Name: {self.name}")
        print(f"Field of Expertise: {self.field_of_expertise}")
        print("Traits:")
        for trait in self.traits:
            print(f"- {trait}")
        print(f"Motivation: {self.motivation}")

    def add_rule(self, condition, action):
        self.decision_maker.add_rule(condition, action)

    def make_decision(self, data):
        return self.decision_maker.make_decision(data)

    def analyze_psychological_state(self, emotion, data):
        return self.psychological_reasoning.analyze(emotion, data)

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

class PsychologicalReasoningEngine:
    def __init__(self):
        self.emotions = {
            'calm': ['analytical', 'observant'],
            'stressed': ['reactive', 'defensive'],
            'curious': ['inquisitive', 'exploratory']
        }

    def analyze(self, emotion, data):
        if emotion in self.emotions:
            traits = self.emotions[emotion]
            return choice(traits)
        else:
            return "Unknown emotion"

# Example usage
hannibal_like_character = Character(
    name="Dr. Smith",
    field_of_expertise="Psychiatry and Culinary Arts",
    traits=["Highly intelligent", "Charismatic", "Manipulative"],
    motivation="A personal philosophy that justifies extreme actions"
)

# Define some rules
hannibal_like_character.add_rule(lambda data: data['threat_level'] > 5, 'neutralize threat')
hannibal_like_character.add_rule(lambda data: data['resource_needed'], 'acquire resources')

# Make a decision
data = {'threat_level': 7, 'resource_needed': False}
decision = hannibal_like_character.make_decision(data)
print(f"Decision: {decision}")

# Analyze psychological state
emotion = 'stressed'
psychological_state = hannibal_like_character.analyze_psychological_state(emotion, data)
print(f"Psychological State: {psychological_state}")

# Display character information
hannibal_like_character.display_character()
```

### Explanation:

1. **Character Class**:
   - **Initialization**: Sets up the character's attributes and initializes a `LogicalDecisionMaker` and a `PsychologicalReasoningEngine`.
   - **Display Character**: Prints out the character's details.
   - **Add Rule**: Adds a rule to the decision-making system.
   - **Make Decision**: Uses the decision-making system to make a decision based on provided data.
   - **Analyze Psychological State**: Uses the psychological reasoning engine to analyze the character's state based on an emotion and data.

2. **LogicalDecisionMaker Class**:
   - **Initialization**: Sets up a dictionary to store rules.
   - **Add Rule**: Adds a rule (condition-action pair) to the dictionary.
   - **Make Decision**: Iterates through the rules and returns the action for the first condition that matches the provided data.

3. **PsychologicalReasoningEngine Class**:
   - **Initialization**: Sets up a dictionary of emotions and associated traits.
   - **Analyze**: Returns a random trait associated with the given emotion.

### Example Usage:

- **Character Creation**: Creates a character named "Dr. Smith" with specific attributes.
- **Rule Definition**: Adds rules for decision-making based on threat level and resource needs.
- **Decision Making**: Makes a decision based on provided data.
- **Psychological Analysis**: Analyzes the psychological state based on an emotion.
- **Display Character Information**: Prints out the character's details.

This structure provides a foundation for a more complex and dynamic character model that can make decisions and analyze its psychological state.