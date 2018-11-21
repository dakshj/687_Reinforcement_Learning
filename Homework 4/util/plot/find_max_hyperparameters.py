import os

import numpy as np

from util.plot.plot_trials import read_stats


def find_max_mean(method_dir):
    dirs = [trials_dir for trials_dir in os.listdir(method_dir)
            if os.path.isdir(os.path.join(method_dir, trials_dir)) and
            'e=' in trials_dir and
            len(os.listdir(os.path.join(method_dir, trials_dir))) >= 100]

    if not dirs:
        print('No directories ready for finding max mean')
        return

    max_means = []
    max_values = []

    for d in dirs:
        results, mean, std, max_mean, max_value = \
            read_stats(trials_dir=os.path.join(method_dir, d))
        max_means.append(max_mean)
        max_values.append(max_value)

    i = int(np.argmax(max_means))
    print('\nDirectory = {}'.format(dirs[i]))
    print('Max Mean = {}'.format(max_means[i]))
    print('Max Value = {}'.format(max_values[i]))


if __name__ == '__main__':
    find_max_mean('../../method/sarsa')
