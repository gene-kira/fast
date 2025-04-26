import random
from datetime import datetime, timedelta
import os

# Automatically load necessary libraries
try:
    from faker import Faker
except ImportError:
    os.system('pip install faker')
    from faker import Faker

fake = Faker()

class Environment:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.characters = []
        self.objects = []
    
    def add_character(self, character):
        self.characters.append(character)
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def describe(self):
        print(f"Welcome to {self.name}.")
        print(self.description)
        if self.characters:
            print("Characters in this environment:")
            for character in self.characters:
                print(f"- {character.name}: {character.description}")
        if self.objects:
            print("Objects in this environment:")
            for obj in self.objects:
                print(f"- {obj.name}")

class Character:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.mood = random.choice(['happy', 'sad', 'angry', 'neutral'])
        self.responses = {
            'happy': ["I'm so happy today!", "It's a beautiful day!"],
            'sad': ["I feel a bit down.", "Not my best day."],
            'angry': ["I'm really frustrated.", "Why is this happening?"],
            'neutral': ["Just another day.", "Nothing much going on."]
        }
    
    def add_response(self, mood, response):
        if mood in self.responses:
            self.responses[mood].append(response)
        else:
            self.responses[mood] = [response]
    
    def interact(self):
        print(f"{self.name}: {random.choice(self.responses[self.mood])}")

class Object:
    def __init__(self, name, description):
        self.name = name
        self.description = description

class User:
    def __init__(self, name):
        self.name = name
        self.environment = None
    
    def enter_environment(self, environment):
        self.environment = environment
        self.environment.describe()
    
    def interact_with_character(self, character_name):
        if not self.environment:
            print("You are not in any environment.")
            return
        
        for character in self.environment.characters:
            if character.name == character_name:
                character.interact()
                break
        else:
            print(f"Character {character_name} is not found in this environment.")

def generate_random_environment():
    name = fake.city()
    description = fake.sentence(nb_words=10)
    environment = Environment(name, description)

    # Add random characters
    for _ in range(random.randint(2, 5)):
        char_name = fake.first_name()
        char_description = fake.sentence(nb_words=6)
        character = Character(char_name, char_description)
        environment.add_character(character)
    
    # Add random objects
    for _ in range(random.randint(1, 3)):
        obj_name = fake.word()
        obj_description = fake.sentence(nb_words=5)
        obj = Object(obj_name, obj_description)
        environment.add_object(obj)

    return environment

def simulate_time_passage(environment):
    # Simulate the passage of time
    for character in environment.characters:
        if random.random() < 0.2:  # 20% chance to change mood
            character.mood = random.choice(['happy', 'sad', 'angry', 'neutral'])

    # Randomly remove or add objects
    if random.random() < 0.1:  # 10% chance to remove an object
        if environment.objects:
            obj_to_remove = random.choice(environment.objects)
            environment.objects.remove(obj_to_remove)
    
    if random.random() < 0.2:  # 20% chance to add a new object
        obj_name = fake.word()
        obj_description = fake.sentence(nb_words=5)
        obj = Object(obj_name, obj_description)
        environment.add_object(obj)

def main():
    user = User("Captain")

    environments = []

    while True:
        print("\nChoose an action:")
        print("1. Enter a new random environment")
        print("2. Change current environment")
        print("3. Interact with characters in the current environment")
        print("4. Simulate time passage in the current environment")
        print("5. Exit")

        choice = input("Enter your choice (1, 2, 3, 4, or 5): ")

        if choice == '1':
            new_environment = generate_random_environment()
            environments.append(new_environment)
            user.enter_environment(new_environment)
        
        elif choice == '2':
            if not environments:
                print("No environments available. Generate a new one.")
            else:
                print("\nChoose an environment to enter:")
                for i, env in enumerate(environments):
                    print(f"{i + 1}. {env.name}")
                
                env_choice = input("Enter your choice (enter the number): ")
                try:
                    env_index = int(env_choice) - 1
                    if 0 <= env_index < len(environments):
                        user.enter_environment(environments[env_index])
                    else:
                        print("Invalid choice. Please choose again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

        elif choice == '3':
            if not user.environment:
                print("You are not in any environment.")
            else:
                while True:
                    print("\nChoose a character to interact with:")
                    for i, character in enumerate(user.environment.characters):
                        print(f"{i + 1}. {character.name}")
                    
                    char_choice = input("Enter your choice (enter the number): ")
                    try:
                        char_index = int(char_choice) - 1
                        if 0 <= char_index < len(user.environment.characters):
                            user.interact_with_character(user.environment.characters[char_index].name)
                            break
                        else:
                            print("Invalid choice. Please choose again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

        elif choice == '4':
            if not user.environment:
                print("You are not in any environment.")
            else:
                simulate_time_passage(user.environment)
                print("\nTime has passed, and the environment has changed:")
                user.environment.describe()

        elif choice == '5':
            break

if __name__ == "__main__":
    main()
