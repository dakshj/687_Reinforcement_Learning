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

    plt.errorbar(range(1, results.shape[1] + 1),
        np.mean(results, axis=0), np.std(results, axis=0), ecolor='yellow')
    text = '{}\n{}\n{}\n{}\n{}'.format(
        'Trials: {}'.format(results.shape[0]),
        'Max Mean:  {}'.format(np.max(np.mean(results, axis=0))),
        'Min Mean:  {}'.format(np.min(np.mean(results, axis=0))),
        'Max Value: {}'.format(np.max(results)),
        'Min Value: {}'.format(np.min(results)),
    )
    plt.gcf().text(0.005, 0.008, text)

    plt.show()
