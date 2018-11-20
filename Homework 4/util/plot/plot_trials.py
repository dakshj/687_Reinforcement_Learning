import os
import time

import matplotlib.pyplot as plt
import numpy as np


def save_trial(arr, trial_dir: str):
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

    np.save('{}/trial_{}'.format(trial_dir, time.time()), arr)


def plot_dir(trials_dir):
    results = np.array(
            [np.load('{}/{}'.format(trials_dir, trial)) for trial in os.listdir(trials_dir)]
    )

    plt.errorbar(range(1, results.shape[1] + 1),
            np.mean(results, axis=0), np.std(results, axis=0), ecolor='yellow')
    text = '{}\n' \
           '{}             {}\n' \
           '{}             {}\n' \
           '{}             {}' \
        .format(
            'Trials: {}'.format(results.shape[0]),
            'Max Mean: {}'.format(np.max(np.mean(results, axis=0))),
            'Min Mean: {}'.format(np.min(np.mean(results, axis=0))),
            'Max Value: {}'.format(np.max(results)),
            'Min Value: {}'.format(np.min(results)),
            'Max Std Dev: {}'.format(np.max(np.std(results, axis=0))),
            'Min Std Dev: {}'.format(np.min(np.std(results, axis=0))),
    )
    plt.gcf().text(0.005, 0.008, text)

    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Expected Returns', fontsize=16)

    plt.show()
