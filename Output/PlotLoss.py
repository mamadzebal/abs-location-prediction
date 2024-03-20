import matplotlib.pyplot as plt


def plot_loss(batch_loss):
    # create an array of indices for the x-axis
    x = range(len(batch_loss))

    # create a line plot of the array values
    plt.plot(x, batch_loss, label = 'Training Loss')

    # display the plot
    plt.show()