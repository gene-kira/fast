import os
import sys
import pygame
from datetime import datetime
import time

class GameHelper:
    def __init__(self):
        self.enemy_ai = EnemyAI()
        self.player = Player()
        self.puzzle_simplifier = PuzzleSimplifier()
        self.challenge_adjuster = ChallengeAdjuster()

    def start_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.event_listener.collect_performance_metrics(event)
            
            # Update game state and performance metrics
            self.update_game_state()

            # Draw real-time feedback on the screen
            self.draw_performance_feedback()

            pygame.display.flip()
            self.clock.tick(30)

    def update_game_state(self):
        self.enemy_ai.adjust_difficulty()
        self.player.apply_modifications()
        self.puzzle_simplifier.simplify_puzzles()
        self.challenge_adjuster.reduce_obstacles()

        # Draw enemy visibility enhancements
        self.enemy_ai.enhance_visibility()
        self.enemy_ai.add_target_lock_on()
        self.enemy_ai.display_health_bars()

    def draw_performance_feedback(self):
        performance = {
            'accuracy': sum([1 for metric in game_state.performance_data if metric['data']['correct']]) / len(game_state.performance_data),
            'time': sum([metric['data']['time'] for metric in game_state.performance_data]),
            'score': sum([metric['data']['score'] for metric in game_state.performance_data])
        }

        accuracy_color = (0, 255, 0) if performance['accuracy'] > 0.8 else (255, 255, 0) if performance['accuracy'] > 0.5 else (255, 0, 0)
        self.screen.fill(accuracy_color, pygame.Rect(10, 60, int(performance['accuracy'] * 780), 30))
        
        time_text = f"Time: {int(performance['time'])} seconds"
        score_text = f"Score: {performance['score']} points"
        
        self.screen.blit(pygame.font.Font(None, 24).render(time_text, True, (255, 255, 255)), (10, 60))
        self.screen.blit(pygame.font.Font(None, 24).render(score_text, True, (255, 255, 255)), (10, 90))

class EnemyAI:
    def __init__(self):
        self.obstacle_density = [1.0, 0.7, 0.5]  # Normal, Easy, Very Easy
        self.current_level = 2

    def adjust_difficulty(self):
        for level in game_state.levels:
            level.obstacle_count *= self.obstacle_density[self.current_level]

    def enhance_visibility(self):
        for enemy in game_state.enemies:
            pygame.draw.circle(self.screen, (255, 0, 0), (enemy.x, enemy.y), 15)  # Red circle around enemies
            pygame.draw.polygon(self.screen, (255, 0, 0), [(enemy.x - 15, enemy.y - 15), (enemy.x + 15, enemy.y - 15), (enemy.x + 15, enemy.y + 15), (enemy.x - 15, enemy.y + 15)])  # Red polygon around enemies

    def add_target_lock_on(self):
        mouse_pos = pygame.mouse.get_pos()
        closest_enemy = min(game_state.enemies, key=lambda e: (e.x - mouse_pos[0])**2 + (e.y - mouse_pos[1])**2)
        if self.target_locked:
            pygame.draw.line(self.screen, (0, 255, 0), mouse_pos, (closest_enemy.x, closest_enemy.y), 3)  # Green line to target

    def display_health_bars(self):
        for enemy in game_state.enemies:
            health_bar_rect = pygame.Rect(enemy.x - 10, enemy.y - 20, 20 * (enemy.health / enemy.max_health), 5)
            pygame.draw.rect(self.screen, (0, 255, 0), health_bar_rect)  # Green health bar
            if enemy.weak_point:
                weak_point_rect = pygame.Rect(enemy.x - 5, enemy.y - 15, 10, 5)
                pygame.draw.rect(self.screen, (255, 0, 0), weak_point_rect)  # Red rectangle for weak point

class Player:
    def __init__(self):
        self.health = 100
        self.max_health = 100
        self.damage_multiplier = 1.2  # All guns do 20% more damage

    def apply_modifications(self):
        # Healing is 100%
        if self.health < self.max_health:
            healing_amount = min(10, self.max_health - self.health)
            self.health += healing_amount

        # Enemy Damage Reduction
        for enemy in game_state.enemies:
            if pygame.sprite.collide_rect(self, enemy):
                damage = enemy.attack * 0.1  # Only take 10% of the damage
                self.health -= damage

    def apply_damage(self, weapon):
        weapon.damage *= self.damage_multiplier  # Increase weapon damage by 20%

class PuzzleSimplifier:
    def __init__(self):
        self.puzzle_steps = {
            'puzzle1': ["Find the key", "Use the key to open the door"],
            'puzzle2': ["Find the hidden button", "Press the button"]
        }
        self.current_puzzle = None

    def simplify_puzzles(self):
        if game_state.current_puzzle in self.puzzle_steps:
            steps = self.puzzle_steps[game_state.current_puzzle]
            for i, step in enumerate(steps):
                hint_text = f"{i + 1}. {step}"
                self.display_hint(hint_text)

    def display_hint(self, text):
        hint_rect = pygame.Rect(10, 130 + (len(self.puzzle_steps[game_state.current_puzzle]) * 30), 780, 50)
        pygame.draw.rect(self.screen, (255, 255, 255), hint_rect, 2)
        self.screen.blit(pygame.font.Font(None, 24).render(text, True, (255, 255, 255)), (10, 130 + (len(self.puzzle_steps[game_state.current_puzzle]) * 30))

class ChallengeAdjuster:
    def __init__(self):
        self.obstacle_density = [1.0, 0.7, 0.5]  # Normal, Easy, Very Easy
        self.current_level = 2

    def reduce_obstacles(self):
        for level in game_state.levels:
            level.obstacle_count *= self.obstacle_density[self.current_level]

class EventListener:
    def __init__(self, enemy_ai, player, puzzle_simplifier, challenge_adjuster):
        self.enemy_ai = enemy_ai
        self.player = player
        self.puzzle_simplifier = puzzle_simplifier
        self.challenge_adjuster = challenge_adjuster

    def collect_performance_metrics(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                hint = self.puzzle_simplifier.display_hint("Explore the environment for hidden items.")
                game_state.performance_data.append({'type': 'hint', 'data': hint})
            elif event.key == pygame.K_s:
                strategy = self.display_strategy()
                game_state.performance_data.append({'type': 'strategy', 'data': strategy})

    def update_performance(self):
        performance = {
            'accuracy': sum([1 for metric in game_state.performance_data if metric['data']['correct']]) / len(game_state.performance_data),
            'time': sum([metric['data']['time'] for metric in game_state.performance_data]),
            'score': sum([metric['data']['score'] for metric in game_state.performance_data])
        }

        self.enemy_ai.adjust_difficulty()
        self.player.apply_modifications()
        self.puzzle_simplifier.simplify_puzzles()
        self.challenge_adjuster.reduce_obstacles()

    def display_strategy(self):
        strategy_text = "Strategy: Focus on collecting coins to increase your score."
        strategy_rect = pygame.Rect(10, 180, 780, 50)
        pygame.draw.rect(game_state.screen, (255, 255, 255), strategy_rect, 2)
        game_state.screen.blit(pygame.font.Font(None, 24).render(strategy_text, True, (255, 255, 255)), (10, 180))
        return {'correct': True, 'time': time.time(), 'score': 10}

def main():
    game_helper = GameHelper()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                game_helper.event_listener.collect_performance_metrics(event)
            
        # Update game state and performance metrics
        game_helper.update_game_state()

        # Draw real-time feedback on the screen
        game_helper.draw_performance_feedback()

if __name__ == "__main__":
    main()
