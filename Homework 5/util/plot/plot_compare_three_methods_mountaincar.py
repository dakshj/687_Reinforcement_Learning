import matplotlib.pyplot as plt

from util.plot.plot_trials import read_trials_dir

dir_1 = '../../method/sarsa_lambda/FINALIZED PLOTS/' \
        'mountaincar__sarsa_lambda__e=0.8__d=0.95__a=0.001__f=4__ep=150__l=0.7'
dir_2 = '../../method/q_lambda/FINALIZED PLOTS/' \
        'mountaincar__q_lambda__e=0.3__d=0.98__a=0.01__f=5__ep=150__l=0.3'
dir_3 = '../../method/actor_critic/FINALIZED PLOTS/' \
        'mountaincar__actor_critic__f=7__l=0.3__aa=0.001__ac=0.05__s=0.9'

results_1, mean_1, std_1, max_mean_1, max_value_1 = \
    read_trials_dir(dir_1, episode_limit=120)
results_2, mean_2, std_2, max_mean_2, max_value_2 = \
    read_trials_dir(dir_2, episode_limit=120)
results_3, mean_3, std_3, max_mean_3, max_value_3 = \
    read_trials_dir(dir_3, episode_limit=120)

plt.errorbar(range(1, results_1.shape[1] + 1), mean_1, std_1, label='Sarsa(λ)',
        color='green', ecolor='lightgreen')

plt.errorbar(range(1, results_2.shape[1] + 1), mean_2, std_2, label='Q(λ)',
        color='blue', ecolor='lightblue')

plt.errorbar(range(1, results_3.shape[1] + 1), mean_3, std_3, label='Actor-Critic',
        color='red', ecolor='lightcoral')

plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Expected Returns', fontsize=16)

plt.title('Mountain Car', fontsize=20)

plt.legend()

plt.show()
