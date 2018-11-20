import os

import numpy as np

from util.plot.plot_trials import read_stats


def find_max_mean(method_dir):
    dirs = [trials_dir for trials_dir in os.listdir(method_dir)
            if os.path.isdir(method_dir) and 'e=' in trials_dir]

    max_means = []
    max_values = []

    for d in dirs:
        results, mean, std, max_mean, max_value = \
            read_stats(trials_dir=os.path.join(method_dir, d))
        max_means.append(max_mean)
        max_values.append(max_value)

    for i, text in zip(
            [int(np.argmax(max_means)), int(np.argmax(max_values))],
            ['Max Mean', 'Max Value']):
        print('\nPicking by {}'.format(text))
        print('Directory = {}'.format(dirs[i]))
        print('Max Mean = {}'.format(max_means[i]))
        print('Max Value = {}'.format(max_values[i]))


if __name__ == '__main__':
    find_max_mean('../../method/sarsa')
