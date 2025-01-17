from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from env import DinoGame
import gymnasium as gym
import pygame

# Step 3: Define the custom callback for logging rewards
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log rewards at the end of each episode
        if "episode" in self.locals:
            self.episode_rewards.append(self.locals["episode"]["r"])
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: Reward: {self.locals['episode']['r']}")
        return True

# Step 1: Initialize the environment
# Note: Replace 'DinoGame-v0' with the actual registered environment if needed.
try:
    env = gym.make("DinoGame-v0")
except gym.error.Error as e:
    print(f"Error: {e}. Make sure 'DinoGame-v0' is a registered Gym environment.")
    exit()

# Step 2: Set up the DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=100000,
    batch_size=32
)

# Step 4: Train the model
timesteps = 800000  # Set the number of training timesteps

reward_callback = RewardLoggerCallback()
model.learn(total_timesteps=timesteps, callback=reward_callback)

# Step 5: Save the model
model_path = "game_model"
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save(os.path.join(model_path, "dqn_dino"))

print(f"Model saved at {os.path.join(model_path, 'dqn_dino.zip')}")
import imageio
from IPython.display import Image

# Initialize the environment
env = gym.make("DinoGame")
obs, _ = env.reset()  # Reset the environment and get the initial observation

frames = []

# Run the game through our learned policy
for _ in range(1000000):
    # Get our action from our learned policy
    action, _ = model.predict(obs, deterministic=True)

    # Take a step in the environment using our action
    obs, reward, done, _, info = env.step(action)

    # Render the current state to a Pygame surface and capture it as a frame
    screen = env.render()
    frame = pygame.surfarray.array3d(screen)
    frame = frame.swapaxes(0, 1)  # Adjust axes for correct orientation
    frames.append(frame)

    # Break the loop if the episode is done
    if done:
        break

# Save the frames as a GIF
gif_path = "game.gif"
imageio.mimsave(gif_path, frames, fps=10)

# Output the GIF in Colab
Image(filename=gif_path)