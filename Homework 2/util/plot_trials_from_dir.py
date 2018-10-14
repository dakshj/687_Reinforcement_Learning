import os

import matplotlib.pyplot as plt
import numpy as np


def plot_dir(trials_dir):
    results = []

    for trial in os.listdir(trials_dir):
        results.append(np.load('{}/{}'.format(trials_dir, trial)))

    results = np.mean(results, axis=0)
    plt.plot(results)
    plt.show()


if __name__ == '__main__':
    plot_dir('../data/gridworld_cross_entropy_trials')
    plot_dir('../data/cartpole_cross_entropy_trials')
