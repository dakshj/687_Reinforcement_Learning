import os

from agent.non_tabular import mountaincar
from agent.non_tabular.mountaincar import MountainCar
from method.sarsa_lambda.sarsa_lambda import sarsa_lambda
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 30

# ALL   = [0.98, 0.95, 0.9, 0.85, 0.8]
EPSILON = [0.98, 0.95, 0.9, 0.85, 0.8]

# `1` means no decay
# ALL         = [0.99, 0.98, 0.95, 0.9]
EPSILON_DECAY = [0.99, 0.98, 0.95, 0.9]

# ALL = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA = [0.1, 0.01, 0.001, 0.005, 0.05]

# ALL               = [3, 4, 5]
FOURIER_BASIS_ORDER = [3, 4, 5]

# ALL    = [150]
EPISODES = [150]

# ALL  = [0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
LAMBDA = [0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

SKIP_EXISTING_PATH = True


def execute():
    for epsilon, epsilon_decay, alpha, fourier_basis_order, episodes, lambda_ in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY,
                    ALPHA, FOURIER_BASIS_ORDER, EPISODES, LAMBDA):
        fourier_basis_order = int(fourier_basis_order)
        episodes = int(episodes)

        trials_dir = '{}__sarsa_lambda__e={}__d={}__a={}__f={}__ep={}__l={}' \
            .format(mountaincar.ENV, epsilon, epsilon_decay,
                alpha, fourier_basis_order, episodes, lambda_)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            agent = MountainCar(fourier_basis_order=fourier_basis_order)
            episode_results = sarsa_lambda(agent=agent,
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes,
                    trials_dir=trials_dir, lambda_=lambda_)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
