import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
DINO_WIDTH, DINO_HEIGHT = 40, 40
DINO_DUCK_HEIGHT = 20
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 40
BIRD_WIDTH, BIRD_HEIGHT = 30, 20
GROUND_HEIGHT = 300
BIRD_HEIGHT_POSITION = GROUND_HEIGHT - 100  # Flying height of the bird
FONT_SIZE = 24
WHITE = (135, 206, 235)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
FPS = 90

class DinoGame(gym.Env):
    def __init__(self):
        super(DinoGame, self).__init__()
        self.state = None
        self.score = 0
        self.action_space = spaces.Discrete(3)  # Action space: 0 = do nothing, 1 = jump, 2 = duck
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(7,),  # [dino_y, dino_ducking, obstacle1_x, obstacle2_x, bird_x, bird_direction, score]
            dtype=np.float32,
        )

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Google Dino Game with Flying Bird")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """ Reset the game to its initial state """
        self.dino_y = GROUND_HEIGHT - DINO_HEIGHT
        self.dino_ducking = False
        self.dino_velocity = 0
        self.is_jumping = False
        self.obstacle1_x = SCREEN_WIDTH
        self.obstacle2_x = SCREEN_WIDTH + random.randint(400, 600)  # Ensure obstacles are farther apart
        self.bird_x = SCREEN_WIDTH // 2
        self.bird_direction = 1     # 1 = moving right, -1 = moving left
        self.bird_speed = 5
        self.obstacle_speed = 4
        self.score = 0

        self.state = np.array(
            [self.dino_y, self.dino_ducking, self.obstacle1_x, self.obstacle2_x, self.bird_x, self.bird_direction, self.score],
            dtype=np.float32,
        )
        return self.state, {}

    def step(self, action):
        """ Update the environment state based on the action taken """
        # Handle jump action
        if action == 1 and not self.is_jumping and not self.dino_ducking:  # Jump if not already jumping or ducking
            self.is_jumping = True
            self.dino_velocity = -16  # Initial upward velocity for the jump

        # Handle duck action
        if action == 2 and not self.is_jumping:  # Duck only when not jumping
            self.dino_ducking = True
        else:
            self.dino_ducking = False

        # Update dino position if jumping
        if self.is_jumping:
            self.dino_y += self.dino_velocity
            self.dino_velocity += 1  # Gravity effect (falling down)
            if self.dino_y >= GROUND_HEIGHT - DINO_HEIGHT:
                self.dino_y = GROUND_HEIGHT - DINO_HEIGHT  # Stop the dino on the ground
                self.is_jumping = False

        # Update obstacle positions
        self.obstacle1_x -= self.obstacle_speed
        self.obstacle2_x -= self.obstacle_speed

        # Reset obstacles if they move off-screen
        if self.obstacle1_x < 0:
            self.obstacle1_x = SCREEN_WIDTH
        if self.obstacle2_x < 0:
            self.obstacle2_x = SCREEN_WIDTH + random.randint(400, 600)  # Ensure obstacles do not overlap

        # Update bird position (moving vertically)
        self.bird_x -= self.bird_speed  # Bird only moves left
        if self.bird_x < 0:  # Reset bird if it goes off-screen
            self.bird_x = SCREEN_WIDTH

        # Check for collisions
        dino_bottom = self.dino_y + (DINO_DUCK_HEIGHT if self.dino_ducking else DINO_HEIGHT)
        dino_right = 50 + DINO_WIDTH
        done = False
        if (
            dino_bottom >= GROUND_HEIGHT - OBSTACLE_HEIGHT and
            ((self.obstacle1_x <= dino_right and self.obstacle1_x + OBSTACLE_WIDTH >= 50) or
             (self.obstacle2_x <= dino_right and self.obstacle2_x + OBSTACLE_WIDTH >= 50))
        ) or (
            self.dino_y <= BIRD_HEIGHT_POSITION + BIRD_HEIGHT and dino_bottom >= BIRD_HEIGHT_POSITION and
            self.bird_x <= dino_right and self.bird_x + BIRD_WIDTH >= 50
        ):
            done = True  # End the game if collision occurs

        # Update score
        self.score += 8 / FPS  # Increment score over time, scaled by FPS

        # Update state and return
        self.state = np.array(
            [self.dino_y, self.dino_ducking, self.obstacle1_x, self.obstacle2_x, self.bird_x, self.bird_direction, self.score],
            dtype=np.float32,
        )
        reward = 1 if not done else -100  # Provide reward, negative if collision
        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        """ Render the game state to the screen """
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, GROUND_HEIGHT), (SCREEN_WIDTH, GROUND_HEIGHT), 2)  # Ground line
        pygame.draw.rect(self.screen, BLACK, (50, self.dino_y, DINO_WIDTH, DINO_DUCK_HEIGHT if self.dino_ducking else DINO_HEIGHT))  # Draw the dino
        pygame.draw.rect(self.screen, RED, (self.obstacle1_x, GROUND_HEIGHT - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))  # Draw obstacle 1
        pygame.draw.rect(self.screen, GREEN, (self.obstacle2_x, GROUND_HEIGHT - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))  # Draw obstacle 2
        pygame.draw.rect(self.screen, BLUE, (self.bird_x, BIRD_HEIGHT_POSITION, BIRD_WIDTH, BIRD_HEIGHT))  # Draw the bird
        score_text = self.font.render(f"Score: {int(self.score)}", True, BLACK)  # Display score
        self.screen.blit(score_text, (10, 10))  # Render score at the top-left corner
        pygame.display.flip()  # Update the display
        self.clock.tick(FPS)  # Control the frame rate

        return self.screen

    def close(self):
        """ Close the Pygame window when done """
        pygame.quit()

# Register the environment
gym.envs.registration.register(
    id='DinoGame-v0',
    entry_point=DinoGame,
)