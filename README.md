# QLearningRobot

Q-learning to learn to correctly pick up cans and avoid walls in his grid world.

Robot lives in a 10 x 10 grid, surrounded by a wall. Some of the grid squares contain soda cans.

It has five “sensors”: Current, North, South, East, and West. 
At any time step, these each return the “value” of the respective location, where the possible values are Empty, Can, and Wall.
It receives a reward of 10 for each can he picks up; a “reward” of −5 if he crashes into a wall (after which he immediately bounces back to the square he was in); and a reward of −1 if he tries to pick up a can in an empty square.
