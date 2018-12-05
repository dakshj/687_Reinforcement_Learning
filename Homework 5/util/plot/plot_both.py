import matplotlib.pyplot as plt

from util.plot.plot_trials import read_trials_dir

dir_1 = '../../method/sarsa_lambda/FINALIZED PLOTS/' \
        'mountaincar__sarsa_lambda__e=0.8__d=0.95__a=0.001__f=4__ep=150__l=0.7'
dir_2 = '../../method/sarsa_lambda/FINALIZED PLOTS/' \
        'mountaincar__sarsa__e=0.4__d=0.95__a=0.005__f=4__ep=150__l=0.0'

results_1, mean_1, std_1, max_mean_1, max_value_1 = \
    read_trials_dir(dir_1, episode_limit=140)
results_2, mean_2, std_2, max_mean_2, max_value_2 = \
    read_trials_dir(dir_2, episode_limit=140)

plt.errorbar(range(1, results_1.shape[1] + 1), mean_1, std_1, ecolor='yellow',
        label='MountainCar Sarsa(Lambda)', color='black')

plt.errorbar(range(1, results_2.shape[1] + 1), mean_2, std_2, ecolor='lightgreen',
        label='MountainCar Sarsa', color='purple')

plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Expected Returns', fontsize=16)

plt.legend()

plt.show()
