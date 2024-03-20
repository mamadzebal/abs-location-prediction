import matplotlib.pyplot as plt


def plot_reward(bst_rwd, time_frames, services, gamma, epsilon_min, lr, type):
    # create an array of indices for the x-axis
    x = range(len(bst_rwd))

    # create a line plot of the array values
    plt.plot(x, bst_rwd)

    # add labels and title
    plt.xlabel('Game')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards for each game')
    
    filename = f"tmp-learning-models/time_frames_{type}_{time_frames}_services_{services}_plot_gamma_{gamma}_epsilon_min_{epsilon_min}_lr_{lr}.png"
    plt.savefig(filename)

    # display the plot
    plt.show()