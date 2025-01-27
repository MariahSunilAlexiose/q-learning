# Interactive Grid World Environment

This project implements a basic and interactive Grid World environment using Python and Pygame, where a robot can navigate a grid using arrow keys. 
The robot encounters various elements within the grid, including:
- **Walls:** Block the robot's movement.
- **Deliveries:** Provide rewards for successful delivery tasks.
- **High-Traffic Zones:** Incur penalties when entered.
- **Terminal State:** The robot reaches its goal, ending the game with a terminal reward.

## Key Features:
- **Grid Customization:** Adjust the grid size and cell size to fit your display preferences.
- **Rewards and Penalties:** The robot collects rewards and incurs penalties based on its movement and interactions within the grid.
- **Visual Representation:** The grid, robot, and goal are visually represented using Pygame, allowing for an interactive experience.

## Customization:
- **Grid Size and Cell Size:** Modify the GRID_SIZE and CELL_SIZE constants to change the grid dimensions.
- **Rewards and Penalties:** Adjust the DELIVERY_REWARD, HIGHTRAFFIC_PENALTY, and TERMINAL_REWARD constants to fine-tune the game dynamics.

## Q-Learning Algorithms:
- **Epsilon-Greedy Strategy:**
  The q_learning_epsilon_greedy function implements the Q-Learning algorithm using the epsilon-greedy strategy. In this approach:
  - The robot selects random actions with probability epsilon to explore the grid.
  - Otherwise, it selects actions with the highest Q-values to exploit known rewards.
  - The Q-values are updated based on the rewards received and the maximum future rewards.
  - The epsilon value decays over time to reduce exploration.
    
- **Exploration with Bonuses and Penalties:**
  The q_learning_exploration function enhances Q-Learning with exploration bonuses and penalties:
    - The robot uses additional exploration values to choose actions that balance exploration and exploitation.
    - The Q-values are updated using the same Q-Learning update rule.
    - The exploration parameter k decays over time to reduce exploration.
