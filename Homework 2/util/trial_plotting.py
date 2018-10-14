import os
import time

import matplotlib.pyplot as plt
import numpy as np


def plot_dir(trials_dir):
    results = []

    for trial in os.listdir(trials_dir):
        results.append(np.load('{}/{}'.format(trials_dir, trial)))

    results = np.mean(results, axis=0)
    plt.plot(results)
    plt.show()


def save_trial(np_arr, trial_dir: str):
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

    np.save('{}/trial_{}'.format(trial_dir, time.time()), np_arr)


if __name__ == '__main__':
    plot_dir('../method/cross_entropy/gridworld_cross_entropy_trials')
    plot_dir('../method/cross_entropy/cartpole_cross_entropy_trials')

    plot_dir('../method/hill_climbing/gridworld_hill_climbing_trials')
    plot_dir('../method/hill_climbing/cartpole_hill_climbing_trials')
