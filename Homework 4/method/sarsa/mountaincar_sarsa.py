import os

from agent.non_tabular import mountaincar
from agent.non_tabular.mountaincar import MountainCar
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL   = [0.1, 0.2, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
EPSILON = [0.8]

# `1` means no decay
# ALL         = [1, 0.99, 0.98, 0.95, 0.9]
EPSILON_DECAY = [0.98]

# ALL = [0.001, 0.005, 0.1, 0.0001, 0.05, 0.02]
ALPHA = [0.1]

# ALL               = [1, 2, 3, 4, 5, 8]
FOURIER_BASIS_ORDER = [5]

# ALL    = [200, 300]
EPISODES = [100]

SKIP_EXISTING_PATH = False


def execute():
    for epsilon, epsilon_decay, alpha, fourier_basis_order, episodes in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY,
                    ALPHA, FOURIER_BASIS_ORDER, EPISODES):
        fourier_basis_order = int(fourier_basis_order)
        episodes = int(episodes)

        trials_dir = '{}__sarsa__e={}__d={}__a={}__f={}__ep={}' \
            .format(mountaincar.ENV, epsilon, epsilon_decay,
                alpha, fourier_basis_order, episodes)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            agent = MountainCar(fourier_basis_order=fourier_basis_order)
            episode_results = sarsa(agent=agent,
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes,
                    trials_dir=trials_dir)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
