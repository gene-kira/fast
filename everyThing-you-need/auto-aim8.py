import math
from typing import List, Tuple
import logging

# Configure basic logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)

class Enemy:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

def calculate_weighted_enemy_position(
    enemies: List[Enemy],
    center_x: float = 0.0,
    center_y: float = 0.0,
    max_distance: float = 50.0,
    weight_multiplier: float = 50.0,
    weighting_strategy: str = "inverse",
    normalization: bool = False
) -> Tuple[float, float]:
    """
    Calculate a weighted average position of enemies relative to a specified center.

    Args:
        enemies (List[Enemy]): List of enemy objects with 'x' and 'y' attributes.
        center_x (float, optional): X-coordinate of the reference center. Defaults to 0.0.
        center_y (float, optional): Y-coordinate of the reference center. Defaults to 0.0.
        max_distance (float, optional): Maximum distance from the center to consider enemies. Defaults to 50.0.
        weight_multiplier (float, optional): Multiplier for calculating weights based on distance. Defaults to 50.0.
        weighting_strategy (str, optional): Strategy to use for weighting ('inverse', 'inverse_square'). Defaults to "inverse".
        normalization (bool, optional): Whether to normalize the weights so they sum to 1. Defaults to False.

    Returns:
        Tuple[float, float]: A tuple containing the weighted average (x, y) position.

    Raises:
        ValueError: If the enemies list is empty or if an enemy lacks 'x' or 'y' attributes.
        TypeError: If invalid types are provided for inputs.
    """
    
    # Input Validation
    if not isinstance(enemies, List[Enemy]):
        raise TypeError("Enemies must be a list of Enemy objects.")
        
    if not enemies:
        raise ValueError("Enemies list cannot be empty.")
        
    if max_distance <= 0 or weight_multiplier <= 0:
        raise ValueError("max_distance and weight_multiplier must be positive numbers.")

    total_mass = 0.0
    total_x = 0.0
    total_y = 0.0
    epsilon = 1e-5  # To avoid division by zero

    for enemy in enemies:
        if not hasattr(enemy, 'x') or not hasattr(enemy, 'y'):
            raise ValueError("Enemy objects must have 'x' and 'y' attributes.")

        x, y = enemy.x, enemy.y
        dx = x - center_x
        dy = y - center_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        logging.debug(f"Processing enemy at ({x}, {y}) with distance {distance} from center.")

        if distance > max_distance:
            logging.debug(f"Enemy at ({x}, {y}) is beyond max_distance; skipping.")
            continue

        # Calculate weight based on the chosen strategy
        if weighting_strategy == "inverse":
            mass = weight_multiplier / (distance + epsilon)
        elif weighting_strategy == "inverse_square":
            mass = weight_multiplier / ((distance + epsilon) ** 2)
        else:
            raise ValueError(f"Invalid weighting strategy: {weighting_strategy}. Choose 'inverse' or 'inverse_square'.")

        total_x += x * mass
        total_y += y * mass
        total_mass += mass

    if total_mass == 0:
        logging.warning("No enemies within max_distance; returning center coordinates.")
        return (center_x, center_y)

    if normalization:
        # Normalize weights so they sum to 1
        avg_x = total_x / total_mass
        avg_y = total_y / total_mass
    else:
        avg_x = total_x / total_mass
        avg_y = total_y / total_mass

    logging.info(f"Weighted average position: ({avg_x}, {avg_y})")

    return (avg_x, avg_y)

# Example usage:
if __name__ == "__main__":
    # Create some example enemies
    enemy1 = Enemy(10.0, 20.0)
    enemy2 = Enemy(-15.0, 25.0)
    enemy3 = Enemy(5.0, 5.0)
    
    enemies = [enemy1, enemy2, enemy3]
    
    # Calculate weighted position with default settings
    weighted_position = calculate_weighted_enemy_position(enemies, center_x=0.0, center_y=0.0)
    print("Weighted Position:", weighted_position)

    # Calculate with different weighting strategy and normalization
    weighted_position_custom = calculate_weighted_enemy_position(
        enemies,
        center_x=0.0,
        center_y=0.0,
        max_distance=50.0,
        weight_multiplier=50.0,
        weighting_strategy="inverse_square",
        normalization=True
    )
    print("Custom Weighted Position:", weighted_position_custom)
