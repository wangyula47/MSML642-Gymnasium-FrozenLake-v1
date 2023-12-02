import gymnasium as gym
import numpy as np
import pygame
from typing import List, Optional
import logging

# Initialize logging
logger = logging.getLogger(__name__)


def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    valid = False

    np.random.seed(seed)

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board.tolist(), size)
    return ["".join(x) for x in board]


# Custom Environment Class for Frozen Lake
class FrozenLakeEnhancedEnv(gym.Env):
    # Metadata for rendering modes
    metadata = {'render.modes': ['human']}

    def __init__(self, size=8, hole_probability=0.2, random_map=False, slippery=True):
        # Initialize the parent class
        super(FrozenLakeEnhancedEnv, self).__init__()

        # Environment parameters
        self.size = size
        self.hole_probability = hole_probability

        # Define start and goal positions
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Four directions
        self.observation_space = gym.spaces.Discrete(size * size)  # Grid size

        # Scalable window size based on environment size
        self.window_size = (min(64 * size, 800), min(64 * size, 800))
        self.cell_size = self.window_size[0] // size
        self.frame_rate = 30  # Default frame rate
        self.current_episode = 0

        # Use random map generation if enabled
        if random_map:
            desc = generate_random_map(size, 1 - hole_probability)
            self.grid = np.array([[1 if cell == 'H' else 0 for cell in row] for row in desc])
        else:
            self.grid = np.zeros((size, size), dtype=int)
            self.place_holes()

        # Set initial state
        self.state = self.start

        # Pygame related initialization
        self.pygame_initialized = False
        self.init_pygame()

        self.slippery = slippery
        self.q_table = None  # Add a placeholder for the Q-table

    def set_episode(self, episode):
        self.current_episode = episode

    # Initialize Pygame for rendering
    def init_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
            self.window_size = (400, 400)
            self.cell_size = self.window_size[0] // self.size
            self.screen = pygame.display.set_mode(self.window_size)
            self.font = pygame.font.Font(None, 24)
            pygame.display.set_caption("Frozen Lake Game")

            # Load images
            self.player_img = pygame.image.load('/Users/wangyu/Desktop/player_image.png')
            self.goal_img = pygame.image.load('/Users/wangyu/Desktop/goal_image.png')
            self.hole_img = pygame.image.load('/Users/wangyu/Desktop/hole_image.png')
            self.ice_img = pygame.image.load('/Users/wangyu/Desktop/ice_image.png')

            # Scale images to cell size
            self.player_img = pygame.transform.scale(self.player_img, (self.cell_size, self.cell_size))
            self.goal_img = pygame.transform.scale(self.goal_img, (self.cell_size, self.cell_size))
            self.hole_img = pygame.transform.scale(self.hole_img, (self.cell_size, self.cell_size))
            self.ice_img = pygame.transform.scale(self.ice_img, (self.cell_size, self.cell_size))

    # Place holes in the grid based on probability
    def place_holes(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != self.start and (i, j) != self.goal:
                    self.grid[i, j] = 1 if np.random.random() < self.hole_probability else 0

    # Step function to update environment's state
    def step(self, action):
        if self.slippery:
            action = self.modify_action_based_on_slipperiness(action)

        current_row, current_col = self.state
        # Update state based on action
        if action == 0:  # left
            current_col = max(0, current_col - 1)
        elif action == 1:  # down
            current_row = min(self.size - 1, current_row + 1)
        elif action == 2:  # right
            current_col = min(self.size - 1, current_col + 1)
        elif action == 3:  # up
            current_row = max(0, current_row - 1)

        self.state = (current_row, current_col)
        done = self.is_hole(self.state) or self.is_goal(self.state)
        reward = self.get_reward(self.state, done)

        return self.state, reward, done, {}

    # Reset the environment to the start state
    def reset(self):
        self.state = self.start
        return self.state

    def modify_action_based_on_slipperiness(self, action):
        # Implement logic to modify action based on slipperiness
        # Example: Randomly choose a different action with a certain probability
        if np.random.rand() < 0.1:  # 10% chance to slip
            return self.action_space.sample()
        return action

    def set_q_table(self, q_table):
        self.q_table = q_table

    # Render the environment on screen
    def render(self, mode='human'):
        self.init_pygame()
        if mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.frame_rate += 5
                    elif event.key == pygame.K_DOWN:
                        self.frame_rate = max(5, self.frame_rate - 5)

            self.screen.fill((255, 255, 255))  # Clear screen

            for i in range(self.size):
                for j in range(self.size):
                    rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                    self.screen.blit(self.ice_img, rect.topleft)

                    if self.grid[i][j] == 1:
                        self.screen.blit(self.hole_img, rect.topleft)

                    if (i, j) == self.goal:
                        self.screen.blit(self.goal_img, rect.topleft)

                    if (i, j) == self.state:
                        self.screen.blit(self.player_img, rect.topleft)

                if self.q_table is not None:
                    for i in range(self.size):
                        for j in range(self.size):
                            state_index = self.state_to_index((i, j))
                            q_values = self.q_table[state_index]
                            self.render_q_values(q_values, i, j)

            # Render text (e.g., state information)
            info_text = self.font.render(f'State: {self.state}', True, (0, 0, 0))
            self.screen.blit(info_text, (5, 5))  # Adjust position as needed

            # UI Enhancements
            fps_text = self.font.render(f'FPS: {self.frame_rate}', True, (0, 0, 0))
            episode_text = self.font.render(f'Episode: {self.current_episode}', True, (0, 0, 0))
            self.screen.blit(fps_text, (5, 20))
            self.screen.blit(episode_text, (5, 40))

            pygame.display.flip()

            pygame.time.wait(1000 // self.frame_rate)

        elif mode == 'ansi':
            # ASCII rendering for non-graphical interface
            output = ""
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) == self.state:
                        output += "P "
                    elif (i, j) == self.goal:
                        output += "G "
                    elif self.grid[i, j] == 1:
                        output += "H "
                    else:
                        output += "F "
                output += "\n"
            print(output)

    def render_q_values(self, q_values, react):
        for action, q_value in enumerate(q_values):
            q_text = f'{action}: {q_value:.2f}'
            text_surface = self.font.render(q_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (react.x, react.y + action * 20))

    def calculate_q_positions(self, rect):
        # Calculate the positions for Q-values once to avoid recalculating them every frame
        q_img = self.font.render(".0000", True, (0, 0, 0))
        q_img_width, q_img_height = q_img.get_width(), q_img.get_height()
        q_positions = [
            (rect.x + self.text_padding, rect.y + self.cell_size // 2),
            (rect.x + self.cell_size // 2 - q_img_width // 2,
             rect.y + self.cell_size - self.text_padding - q_img_height),
            # ... [Other positions]
        ]
        return q_positions

    def _render_text(self):
        # Enhanced ANSI rendering for terminal display
        output = ""
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.state:
                    output += "P "
                elif (i, j) == self.goal:
                    output += "G "
                elif self.grid[i, j] == 1:
                    output += "H "
                else:
                    output += "F "
            output += "\n"
        print(output)

    # Helper functions for game logic
    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def is_hole(self, position):
        return self.grid[position] == 1

    def is_goal(self, position):
        return position == self.goal

    def get_reward(self, position, done):
        if self.is_goal(position):
            return 1
        if self.is_hole(position):
            return -1
        return 0
