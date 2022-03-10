from Robot import Robot
import matplotlib.pyplot as plt


def plot(rewards_list, test=False):
    if test:
        plt.title("Test Data")
    plt.plot(rewards_list[0], linestyle='dotted')  # average reward value for each episode in the last 100 episodes
    plt.ylabel("average rewards")
    plt.xlabel("episodes in 100")
    plt.show()
    if test:
        plt.title("Test Data")
    plt.plot(rewards_list[1], linestyle='solid')  # rewards captured at each 100th episode
    plt.ylabel("rewards")
    plt.xlabel("episodes in 100")
    plt.show()

num_episodes = int(input("Please enter number of episodes:\n"))
max_steps_per_episode = int(input("Please enter number of steps:\n"))
epsilon = 0.1
learning_rate = 0.2
discount_rate = 0.9
q_matrix = {}
robot = Robot()
rewards = robot.train(num_episodes, max_steps_per_episode, q_matrix, epsilon, learning_rate, discount_rate)
plot(rewards)
rewards1 = robot.test(num_episodes, max_steps_per_episode, q_matrix, 0, learning_rate, discount_rate)
plot(rewards1, True)
