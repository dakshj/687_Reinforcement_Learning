import os

from agent.non_tabular import mountaincar
from agent.non_tabular.mountaincar import MountainCar
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL   = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
EPSILON = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]

# `1` means no decay
# ALL         = [1, 0.98, 0.9, 0.85]
EPSILON_DECAY = [1, 0.98, 0.9, 0.85]

# BAD = [0.1]
# ALL = [0.000001, 0.00001, 0.0001, 0.00005, 0.001, 0.0005, 0.1]
ALPHA = [0.001, 0.0005, 0.1, 0.000001, 0.0001]

# ALL               = [3, 4, 5]
FOURIER_BASIS_ORDER = [3, 4, 5]

# ALL    = [100, 200]
EPISODES = [300]

SKIP_EXISTING_PATH = True


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
