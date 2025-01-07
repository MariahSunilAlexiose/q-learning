# -*- coding: utf-8 -*-
"""
This code provides a basic and interactive grid world environment where a robot can navigate 
using the arrow keys. The robot encounters walls that block movement, rewards for successful deliveries, 
and penalties for high-traffic zones. The game ends when the robot reaches its terminal state. 
The robot's score reflects the rewards it collects and penalties it incurs.
"""

import pygame
import numpy as np
import random

# Constants for our display
GRID_SIZE = 10  # Easily change this value
CELL_SIZE = 50  # Adjust this based on your display preferences
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
DELIVERY_REWARD = 20 
HIGHTRAFFIC_PENALTY = -10 
TERMINAL_REWARD = 100
STEP_PENALTY = -2  # Constant step penalty
ROBOT_COLOR = (0, 128, 255)
GOAL_COLOR = (0, 255, 0)
WALL_COLOR = (0, 0, 0)
EMPTY_COLOR = (255, 255, 255)
DELIVERY_COLOR = (255, 255, 0)  # Yellow for delivery
HIGHTRAFFIC_COLOR = (255, 0, 0)   # Red for high-traffic

random.seed(100)

class GridWorld:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size))
        # Randomly select start and goal positions
        self.start = (random.randint(0, size-1), random.randint(0, size-1))
        self.goal = (random.randint(0, size-1), random.randint(0, size-1))
        self.robot_pos = self.start
        self.score = 0

        # Variables needed for Q Learning
        self.episodes = 25
        self.actions = ["up", "down", "left", "right"]
        self.q_table = np.zeros((self.size, self.size, len(self.actions)))
        self.visit_count = np.zeros((self.size, self.size, len(self.actions)))
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.alpha = 0.9
        self.gamma = 0.9

        self.generate_walls_traffic_delivery()

    def generate_walls_traffic_delivery(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != self.start and (i, j) != self.goal:
                    rand_num = random.random()
                    if rand_num < 0.2:  # 20% chance for a wall
                        self.grid[i][j] = np.inf
                    elif rand_num < 0.4:  # 20% chance for delivery reward
                        self.grid[i][j] = DELIVERY_REWARD
                    elif rand_num > 0.9:  # 20% chance for high-traffic penalty
                        self.grid[i][j] = HIGHTRAFFIC_PENALTY

    def move(self, direction):
        """Move the robot in a given direction."""
        x, y = self.robot_pos
        initial_score = self.score

        # Conditions check for boundaries and walls
        if direction == "up" and x > 0 and self.grid[x-1][y] != np.inf:
            x -= 1
        elif direction == "down" and x < self.size-1 and self.grid[x+1][y] != np.inf:
            x += 1
        elif direction == "left" and y > 0 and self.grid[x][y-1] != np.inf:
            y -= 1
        elif direction == "right" and y < self.size-1 and self.grid[x][y+1] != np.inf:
            y += 1

        # Update the robot's position
        self.robot_pos = (x, y)
        
        # Add a penalty for revisiting cells
        if self.visit_count[x, y, :].any():
            reward = self.grid[x, y] + HIGHTRAFFIC_PENALTY

        # Add the specific reward or penalty based on the cell value
        if self.grid[x, y] == DELIVERY_REWARD:
            reward = self.grid[x, y] + DELIVERY_REWARD 
        elif self.grid[x, y] == HIGHTRAFFIC_PENALTY:
            reward = self.grid[x, y] + HIGHTRAFFIC_PENALTY
        elif self.robot_pos == self.goal:
            reward = TERMINAL_REWARD  # Terminal state reward
            print("Robot reached the terminal state!")
            print(f"Final Score: {self.score}")
            exit()
        else:
            reward = self.grid[x][y] + STEP_PENALTY  # Add step penalty to any existing cell reward/penalty

        # Update the grid, score, and return the reward
        self.grid[x][y] = 0  # Clear the cell after the robot moves
        self.score += reward

        # Ensure the score does not become infinite
        if np.isinf(self.score) or np.isnan(self.score):
            self.score = initial_score    # Reset to initial score to prevent issues

        return reward

    def display(self):
        """Print a text-based representation of the grid world (useful for debugging)."""
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                if (i, j) == self.robot_pos:
                    row += 'R '
                elif np.array_equal(self.grid[i][j], WALL_COLOR):
                    row += '# '
                elif np.array_equal(self.grid[i][j], DELIVERY_COLOR):
                    row += 'D '
                elif np.array_equal(self.grid[i][j], HIGHTRAFFIC_COLOR):
                    row += 'H '
                else:
                    row += '. '
            print(row)

    def q_learning_epsilon_greedy(self, screen):
        for episode in range(self.episodes):           
            self.robot_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))    # Randomize start position
            cumulative_reward = 0            
            
            for i in range(self.size):
                for j in range(self.size):
                    if np.random.rand() < self.epsilon:                       
                        action_index = np.random.randint(0, len(self.actions))    # Picks a random action from the possible actions
                    else:
                        action_index = np.argmax(self.q_table[i, j])    # Exploit
                    
                    action = self.actions[action_index]    # Selects the corresponding action                  
                    reward = self.move(action)    # Moves the robot
                    ni, nj = self.robot_pos

                    # Check if the new position is out of bounds or blocked (wall)
                    if ni < 0 or nj < 0 or ni >= self.size or nj >= self.size or self.grid[ni][nj] == np.inf:
                        reward -= 1
                        self.robot_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))    # Reset to a random position

                    # Ensure the reward does not become infinite
                    if np.isinf(reward) or np.isnan(reward):
                        reward = STEP_PENALTY    # Set a default step penalty

                    sample = reward + self.gamma * np.max(self.q_table[ni, nj])    # Calculates the sample using the reward and the maximum Q-value of the next state
                    self.q_table[i, j, action_index] = (1 - self.alpha) * self.q_table[i, j, action_index] + self.alpha * sample # Update Q-value using the Q-learning update rule (weighted average)
                    self.visit_count[i, j, action_index] += 1    # Update visit count
                    cumulative_reward += reward

                    # Ensure the cumulative reward does not become infinite
                    if np.isinf(cumulative_reward) or np.isnan(cumulative_reward):
                        cumulative_reward = 0    # Reset to zero to prevent issues

                    # Update the screen
                    screen.fill(EMPTY_COLOR)
                    draw_grid(self, screen)
                    pygame.display.flip()
                    pygame.time.delay(10)    # delay in ms
                    
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)    # Decay epsilon
            print(f"Episode {episode + 1}, Cumulative Reward: {cumulative_reward}")

def setup_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid World")
    clock = pygame.time.Clock()
    return screen, clock

def draw_grid(world, screen):
    """Render the grid, robot, and goal on the screen."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):            
            color = EMPTY_COLOR
            cell_value = world.grid[i][j]
            if cell_value == np.inf:
                color = WALL_COLOR
            elif cell_value == DELIVERY_REWARD:  
                color = DELIVERY_COLOR
            elif cell_value == HIGHTRAFFIC_PENALTY:  
                color = HIGHTRAFFIC_COLOR
            pygame.draw.rect(screen, color, pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Drawing the grid lines
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_HEIGHT))
        pygame.draw.line(screen, (200, 200, 200), (0, i * CELL_SIZE), (SCREEN_WIDTH, i * CELL_SIZE))

    pygame.draw.circle(screen, ROBOT_COLOR, 
                       (int((world.robot_pos[1] + 0.5) * CELL_SIZE), int((world.robot_pos[0] + 0.5) * CELL_SIZE)), 
                       int(CELL_SIZE/3))

    # Draw the goal as a rectangle
    pygame.draw.rect(screen, GOAL_COLOR, 
                     pygame.Rect(world.goal[1] * CELL_SIZE, world.goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def main():
    """Main loop"""
    screen, clock = setup_pygame()
    world = GridWorld()
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Move robot based on arrow key press
                if event.key == pygame.K_UP:
                    world.move("up")
                if event.key == pygame.K_DOWN:
                    world.move("down")
                if event.key == pygame.K_LEFT:
                    world.move("left")
                if event.key == pygame.K_RIGHT:
                    world.move("right")
                # Print the score after the move
                print(f"Current Score: {world.score}")
                # Check if the robot reached the goal
                if world.robot_pos == world.goal:
                    print(f"Final Score: {world.score}")
                    running = False
                    break
        
        world.q_learning_epsilon_greedy(screen)

        # Rendering
        screen.fill(EMPTY_COLOR)
        draw_grid(world, screen)
        pygame.display.flip()
        
        clock.tick(10)  # FPS

    pygame.quit()

if __name__ == "__main__":
    main()