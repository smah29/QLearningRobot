import random
import Grid
import numpy as np
import statistics
pickup = 0
north = 1
south = 2
east = 3
west = 4


def select_action(q_vales, epsilon):
    if (epsilon * 1000) >= random.randint(1, 1000):
        return random.randint(0, 4)
    else:
        return np.argmax(q_vales)


class Robot:

    def __init__(self):
        self.x = 1
        self.y = 1
        self.exploration_decay_rate = 0.001

    def sense_curr(self, grid):
        return grid[self.x][self.y]

    def sense_north(self, grid):
        return grid[self.x][self.y + 1]

    def sense_south(self, grid):
        return grid[self.x][self.y - 1]

    def sense_east(self, grid):
        return grid[self.x + 1][self.y]

    def sense_west(self, grid):
        return grid[self.x - 1][self.y]

    def pick_up(self, grid):
        if grid[self.x][self.y] == Grid.can:
            grid[self.x][self.y] = Grid.blank
            return True
        else:
            return False

    def move_north(self, grid):
        if self.sense_north(grid) == Grid.wall:
            return False
        self.y += 1
        return True

    def move_south(self, grid):
        if self.sense_south(grid) == Grid.wall:
            return False
        self.y -= 1
        return True

    def move_east(self, grid):
        if self.sense_east(grid) == Grid.wall:
            return False
        self.x += 1
        return True

    def move_west(self, grid):
        if self.sense_west(grid) == Grid.wall:
            return False
        self.x -= 1
        return True

    def perform_action(self, action, grid):
        if action == pickup:
            success = self.pick_up(grid)
            if success:
                return 10
            else:
                return -1
        elif action == north:
            success = self.move_north(grid)
            if success:
                return 0
            else:
                return -5
        elif action == south:
            success = self.move_south(grid)
            if success:
                return 0
            else:
                return -5
        elif action == east:
            success = self.move_east(grid)
            if success:
                return 0
            else:
                return -5
        elif action == west:
            success = self.move_west(grid)
            if success:
                return 0
            else:
                return -5

    def get_state(self, grid):  # represents state as a tuple
        state_vector = (
            self.sense_curr(grid), self.sense_north(grid), self.sense_south(grid),
            self.sense_east(grid), self.sense_west(grid))
        return state_vector

    def generate_robot_initial_position(self):
        self.x = random.randint(1, 10)
        self.y = random.randint(1, 10)

    def step(self, grid, q_matrix, learning_rate, discount_rate, epsilon, training):
        curr_state = self.get_state(grid)
        if curr_state not in q_matrix:  # if first time seeing state then add to q_matrix
            q_matrix[curr_state] = np.zeros(5)
        action = select_action(q_matrix[curr_state], epsilon)
        reward = self.perform_action(action, grid)
        new_state = self.get_state(grid)
        if new_state not in q_matrix:
            q_matrix[new_state] = np.zeros(5)
        if training:
            q_matrix[curr_state][action] = q_matrix[curr_state][action] + learning_rate * (
                    reward + discount_rate * max(q_matrix[new_state]) - q_matrix[curr_state][action])
        return reward

    def episode(self, max_steps_per_episode, grid, q_matrix, learning_rate, discount_rate, epsilon, training):
        current_episode_reward = 0
        for step in range(0, max_steps_per_episode):
            current_episode_reward += self.step(grid, q_matrix, learning_rate, discount_rate, epsilon, training)
        return current_episode_reward

    def train(self, num_episodes, max_steps_per_episode, q_matrix, epsilon, learning_rate, discount_rate):
        rewards_each_100th_episodes = []
        avg_rewards_each_100th_episodes = []
        rewards_sum = 0
        for episode in range(num_episodes):
            self.generate_robot_initial_position()
            grid = Grid.create_grid()
            if (episode + 1) % 50 == 0 and epsilon != 0.0:
                epsilon -= self.exploration_decay_rate
            current_episode_reward = self.episode(max_steps_per_episode, grid,
                                                  q_matrix, learning_rate, discount_rate, epsilon, True)
            rewards_sum += current_episode_reward
            if (episode + 1) % 100 == 0:
                avg_rewards_each_100th_episodes.append(rewards_sum / 100)
                rewards_each_100th_episodes.append(current_episode_reward)
                rewards_sum = 0
        return [avg_rewards_each_100th_episodes, rewards_each_100th_episodes]

    def test(self, num_episodes, max_steps_per_episode, q_matrix, epsilon, learning_rate, discount_rate):
        rewards_each_100th_episodes = []
        avg_rewards_each_100th_episodes = []
        reward_total = 0
        rewards_sum = 0
        reward_vals = []
        for episode in range(num_episodes):
            self.generate_robot_initial_position()
            grid = Grid.create_grid()
            current_episode_reward = self.episode(max_steps_per_episode, grid,
                                                  q_matrix, learning_rate, discount_rate, epsilon, False)
            rewards_sum += current_episode_reward
            reward_total += current_episode_reward
            reward_vals.append(current_episode_reward)
            if (episode + 1) % 100 == 0:
                avg_rewards_each_100th_episodes.append(rewards_sum / 100)
                rewards_each_100th_episodes.append(current_episode_reward)
                rewards_sum = 0
        print("---------Test Results---------")
        print("Average reward value: " + str(reward_total / num_episodes))
        print("Standard deviation: " + str(statistics.pstdev(reward_vals)))
        return [avg_rewards_each_100th_episodes, rewards_each_100th_episodes]
