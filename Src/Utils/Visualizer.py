import matplotlib.pyplot as plt
import seaborn as sns
import os


def Visualizer():
    print("Visualizer")


def ScatterPlot(X, Y, title, xlabel, ylabel, color, path):
    plt.scatter(X, Y, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(os.path.join(path, "Scatter_"+title + ".png"))
