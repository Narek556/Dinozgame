Explanation of the Game
This is a custom implementation of the Google Dino game using Gymnasium and Pygame, with the added challenge of a flying bird. The goal is to avoid obstacles (ground-based and airborne) while earning points over time. The game features a jumping and ducking mechanism for the Dino, as well as dynamic obstacles.

1.State
The game state is represented as a 7-dimensional vector:

dino_y: The vertical position of the Dino.
dino_ducking: A binary flag (True/False) indicating whether the Dino is ducking.
obstacle1_x: The x-position of the first ground obstacle.
obstacle2_x: The x-position of the second ground obstacle.
bird_x: The x-position of the flying bird.
bird_direction: Not used currently but included in the state (default is 1 for movement to the left).
score: The player's cumulative score.

2.Actions
The player can choose from three discrete actions:

0: Do nothing (the Dino keeps its current state).
1: Jump (the Dino jumps upward if not already jumping or ducking).
2: Duck (the Dino ducks if not already jumping).

3.Reward
+1: Awarded for each step without collision.
-100: Issued when the Dino collides with an obstacle or the bird, ending the game.

4.Contributions
Developed at TUMO.
Built using:
Gymnasium: For creating the custom game environment.
Stable-Baselines3: For training the reinforcement learning agent.
Pygame: For rendering the game.

5.Changes/Notes about the Dino
The Dino's physics include a jump mechanic with gravity and a duck mechanic that reduces its height for avoiding obstacles and the bird.
The bird is positioned at a constant height but moves leftward. Future improvements could involve dynamic vertical bird movement for increased difficulty.












