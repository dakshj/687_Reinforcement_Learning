import os
import time

import matplotlib.pyplot as plt
import numpy as np


def save_trial(np_arr, trial_dir: str):
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

    np.save('{}/trial_{}'.format(trial_dir, time.time()), np_arr)


def plot_dir(trials_dir):
    results = np.array(
        [np.load('{}/{}'.format(trials_dir, trial)) for trial in os.listdir(trials_dir)]
    )

    print('shape = {}'.format(results.shape))
    print('max = {}'.format(np.max(np.mean(results, axis=0))))

    plt.errorbar(range(1, results.shape[1] + 1),
        np.mean(results, axis=0), np.std(results, axis=0), ecolor='yellow')

    plt.show()


if __name__ == '__main__':
    plot_dir('../method/cross_entropy/gridworld_cross_entropy_trials')
    plot_dir('../method/cross_entropy/cartpole_cross_entropy_trials')

    plot_dir('../method/hill_climbing/gridworld_hill_climbing_trials')
    plot_dir('../method/hill_climbing/cartpole_hill_climbing_trials')
