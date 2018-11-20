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

    print('Directory = {}'.format(dirs[int(np.argmax(max_means))]))
    print('Max Mean = {}'.format(np.max(max_means)))
    print('Max Value = {}'.format(max_values[int(np.argmax(max_means))]))


if __name__ == '__main__':
    find_max_mean('../../method/sarsa')

# RESULTS:
# Directory = cartpole__sarsa__e=0.3__a=0.0001__f=3
# Max Mean = 77.46
# Max Value = 281.0
