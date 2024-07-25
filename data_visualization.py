import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def plot_histogram(array, x, index, pl):
    color = ['red', 'yellow']
    for i in range(len(index) - 1):
        house = array[index[i]:index[i + 1] , x]
        pl.hist(house, color=color[i], alpha=0.5)

def scatter_plot(data, x, y, index, pl):
    color = ['red', 'yellow']
    i = 0
    for i in range(len(index) - 1):
        housex = data[index[i]:index[i + 1] , x]
        housey = data[index[i]:index[i + 1] , y]
        pl.scatter(housex, housey, s=1, color=color[i], alpha=0.5)
        i += 1


def all_plot(data, result):
    size = len(data.T)
    _, ax= plt.subplots(nrows=size, ncols=size)
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    change_indices = np.where(result[:-1] != result[1:])[0] + 1
    index = np.concatenate(([0], change_indices, [len(data)]))
    for y in range(0, size):
        for x in range(0, size):
            if(y == x):
                plot_histogram(data, x, index, ax[y, x])
            else:
                scatter_plot(data, x, y, index, ax[y, x])
            ax[y, x].tick_params(labelbottom=False)
            ax[y, x].tick_params(labelleft=False)

            ax[y, x].spines['right'].set_visible(False)
            ax[y, x].spines['top'].set_visible(False)

def getData(path):
    df = pd.read_csv(path)
    data = np.array(df)
    return data[:, 1:]
def visualize(data):
    sorted_index = np.argsort(data[:, 0], axis=0)
    sorted_array = data[sorted_index]
    all_plot(np.array(sorted_array[:, 1:], dtype=float), sorted_array[:, :1])
    plt.show()
    
def main():
    data = getData(sys.argv[1])
    visualize(data)
if __name__ == "__main__":
    main()
