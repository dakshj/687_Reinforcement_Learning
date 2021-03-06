import os
import time

import matplotlib.pyplot as plt
import numpy as np


def save_trial(arr, trials_dir: str):
    if not os.path.exists(trials_dir):
        os.mkdir(trials_dir)

    np.save('{}/trial_{}'.format(trials_dir, time.time()), arr)


def read_stats(trials_dir):
    results = np.array(
            [np.load('{}/{}'.format(trials_dir, trial)) for trial in os.listdir(trials_dir)]
    )

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    max_mean = np.max(np.mean(results, axis=0))
    max_value = np.max(results)

    return results, mean, std, max_mean, max_value


def plot_dir(trials_dir):
    results, mean, std, max_mean, max_value = read_stats(trials_dir)

    plt.errorbar(range(1, results.shape[1] + 1), mean, std, ecolor='yellow')
    text = '{}\n' \
           '{}             {}\n' \
           '{}             {}\n' \
           '{}             {}' \
        .format(
            'Trials: {}'.format(results.shape[0]),
            'Max Mean: {}'.format(max_mean),
            'Min Mean: {}'.format(np.min(np.mean(results, axis=0))),
            'Max Value: {}'.format(max_value),
            'Min Value: {}'.format(np.min(results)),
            'Max Std Dev: {}'.format(np.max(np.std(results, axis=0))),
            'Min Std Dev: {}'.format(np.min(np.std(results, axis=0))),
    )
    plt.gcf().text(0.005, 0.008, text)

    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Expected Returns', fontsize=16)

    plt.show()


if __name__ == '__main__':
    plot_dir('../../method/sarsa/'
             'cartpole__sarsa_tc__e=0.95__d=1.0__ep=150__t1=8__t2=20')
