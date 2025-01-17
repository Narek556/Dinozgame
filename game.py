from env import DinoGame
import pygame
if __name__ == "__main__":
    env = DinoGame()
    done = False
    obs, _ = env.reset()

    print("Press SPACE to jump, DOWN arrow to duck. Close the game window to exit.")

    while not done:
        env.render()

        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2

        obs, reward, done, _, _ = env.step(action)
        print(f"Score: {int(obs[6])}, Dino Y: {obs[0]}, Bird X: {obs[4]}, Bird Direction: {obs[5]}")

    env.close()