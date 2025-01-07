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
- **Q-Learning Algorithm:** The robot uses Q-learning with epsilon-greedy strategy to learn and improve its navigation over multiple episodes.
- **Visual Representation:** The grid, robot, and goal are visually represented using Pygame, allowing for an interactive experience.

## Customization:
- **Grid Size and Cell Size:** Modify the GRID_SIZE and CELL_SIZE constants to change the grid dimensions.
- **Rewards and Penalties:** Adjust the DELIVERY_REWARD, HIGHTRAFFIC_PENALTY, and TERMINAL_REWARD constants to fine-tune the game dynamics.
