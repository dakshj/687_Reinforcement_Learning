import os

from agent import cartpole
from agent.cartpole import CartPole
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# OTHERS = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
EPSILON = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# BAD = [0.1]
# OTHERS = [0.000001, 0.00001, 0.0001, 0.00005, 0.001, 0.0005]
ALPHA = [0.000001, 0.00001, 0.00005, 0.0005]

# OTHERS = [3, 4, 5]
FOURIER_BASIS_ORDER = [3, 4, 5]

# OTHERS = [100, 200]
EPISODES = [200]


def execute():
    for epsilon, alpha, fourier_basis_order, episodes in \
            random_hyperparameter_search(EPSILON, ALPHA, FOURIER_BASIS_ORDER, EPISODES):
        fourier_basis_order = int(fourier_basis_order)
        episodes = int(episodes)

        trials_dir = '{}__sarsa__e={}__a={}__f={}__ep={}' \
            .format(cartpole.ENV, epsilon, alpha, fourier_basis_order, episodes)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        skip_existing_path = True
        if skip_existing_path and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            agent = CartPole(epsilon=epsilon, fourier_basis_order=fourier_basis_order)
            episode_results = sarsa(agent=agent, alpha=alpha,
                    trial=trial, trials_total=TRIALS, episodes=episodes)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
