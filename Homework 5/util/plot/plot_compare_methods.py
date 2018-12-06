import matplotlib.pyplot as plt

from util.plot.plot_trials import read_trials_dir

dir_1 = '../../method/sarsa_lambda/FINALIZED PLOTS/' \
        'gridworld__sarsa_lambda__e=0.9__d=0.95__a=0.2__ep=150__l=0.8'
dir_2 = '../../method/q_lambda/FINALIZED PLOTS/' \
        'gridworld__q_lambda__e=0.98__d=0.95__a=0.3__ep=150__l=0.1'
dir_3 = '../../method/actor_critic/FINALIZED PLOTS/' \
        'gridworld__actor_critic__l=0.9__aa=0.1__ac=0.1__s=0.9'

results_1, mean_1, std_1, max_mean_1, max_value_1 = \
    read_trials_dir(dir_1, episode_limit=140)
results_2, mean_2, std_2, max_mean_2, max_value_2 = \
    read_trials_dir(dir_2, episode_limit=140)
results_3, mean_3, std_3, max_mean_3, max_value_3 = \
    read_trials_dir(dir_3, episode_limit=140)

plt.errorbar(range(1, results_1.shape[1] + 1), mean_1, std_1, label='Sarsa(λ)',
        color='green', ecolor='lightgreen')

plt.errorbar(range(1, results_2.shape[1] + 1), mean_2, std_2, label='Q(λ)',
        color='blue', ecolor='lightblue')

plt.errorbar(range(1, results_3.shape[1] + 1), mean_3, std_3, label='Actor-Critic',
        color='red', ecolor='lightcoral', )

plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Expected Returns', fontsize=16)

plt.title('GridWorld', fontsize=20)

plt.legend()

plt.show()
