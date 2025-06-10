

```python
from random import choice

class LogicalDecisionMaker:
    def __init__(self):
        self.rules = {}
        self.psychological_rules = {}

    def add_rule(self, condition, action):
        self.rules[condition] = action

    def add_psychological_reasoning(self, emotions, actions):
        for emotion in emotions:
            if emotion not in self.psychological_rules:
                self.psychological_rules[emotion] = []
            self.psychological_rules[emotion].extend(actions)

    def make_decision(self, data, current_emotion=None):
        # Logical decision-making
        for condition, action in self.rules.items():
            if condition(data):
                return action
        
        # Psychological reasoning
        if current_emotion and current_emotion in self.psychological_rules:
            return choice(self.psychological_rules[current_emotion])
        
        return None

# Example usage:
decision_maker = LogicalDecisionMaker()

# Define some logical rules
decision_maker.add_rule(lambda data: data['threat_level'] > 5, 'neutralize threat')
decision_maker.add_rule(lambda data: data['resource_needed'], 'acquire resources')

# Define some psychological reasoning
emotions = ['happy', 'angry', 'sad']
actions_happy = ['celebrate', 'share success']
actions_angry = ['attack', 'retaliate']
actions_sad = ['withdraw', 'seek comfort']

decision_maker.add_psychological_reasoning(emotions, actions_happy)
decision_maker.add_psychological_reasoning(emotions, actions_angry)
decision_maker.add_psychological_reasoning(emotions, actions_sad)

# Make a decision
data = {'threat_level': 7, 'resource_needed': False}
current_emotion = 'angry'
decision = decision_maker.make_decision(data, current_emotion)
print(decision)  # Output: attack or retaliate (randomly chosen)
```

### Character Class Inspired by Hannibal Lecter

Next, let's enhance the character class to include more detailed attributes and methods that reflect a complex personality.

```python
class Character:
    def __init__(self, name, field_of_expertise, traits, motivation, current_emotion='neutral'):
        self.name = name
        self.field_of_expertise = field_of_expertise
        self.traits = traits
        self.motivation = motivation
        self.current_emotion = current_emotion
        self.actions = []

    def display_character(self):
        print(f"Name: {self.name}")
        print(f"Field of Expertise: {self.field_of_expertise}")
        print("Traits:")
        for trait in self.traits:
            print(f"- {trait}")
        print(f"Motivation: {self.motivation}")
        print(f"Current Emotion: {self.current_emotion}")

    def set_emotion(self, emotion):
        self.current_emotion = emotion

    def add_action(self, action):
        self.actions.append(action)

    def perform_action(self, data, decision_maker):
        decision = decision_maker.make_decision(data, self.current_emotion)
        if decision:
            print(f"{self.name} decides to {decision}.")
            self.add_action(decision)
        else:
            print(f"{self.name} is indecisive.")

# Example usage
hannibal_like_character = Character(
    name="Dr. Smith",
    field_of_expertise="Psychiatry and Culinary Arts",
    traits=["Highly intelligent", "Charismatic", "Manipulative"],
    motivation="A personal philosophy that justifies extreme actions"
)

decision_maker = LogicalDecisionMaker()
decision_maker.add_rule(lambda data: data['threat_level'] > 5, 'neutralize threat')
decision_maker.add_rule(lambda data: data['resource_needed'], 'acquire resources')

emotions = ['happy', 'angry', 'sad']
actions_happy = ['celebrate', 'share success']
actions_angry = ['attack', 'retaliate']
actions_sad = ['withdraw', 'seek comfort']

decision_maker.add_psychological_reasoning(emotions, actions_happy)
decision_maker.add_psychological_reasoning(emotions, actions_angry)
decision_maker.add_psychological_reasoning(emotions, actions_sad)

hannibal_like_character.display_character()
hannibal_like_character.set_emotion('angry')
data = {'threat_level': 7, 'resource_needed': False}
hannibal_like_character.perform_action(data, decision_maker)
```

