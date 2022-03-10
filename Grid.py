import numpy as np
blank: int = 0
can = 1
wall = 2


def create_grid():
    grid = np.random.randint(2, size=(12, 12))
    for i, g in enumerate(grid):
        for j, gr in enumerate(grid[i]):
            if j == 0 or j == 11 or i == 0 or i == 11:
                grid[i][j] = wall
    return grid
