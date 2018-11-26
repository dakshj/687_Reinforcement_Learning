import os

from agent.non_tabular import cartpole
from agent.non_tabular.cartpole import CartPole
from method.q_learning.q_learning import q_learning
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL   = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
EPSILON = [0.7]

# `1` means no decay
# ALL         = [1, 0.99, 0.98, 0.95, 0.9, 0.85]
EPSILON_DECAY = [0.95]

# ALL = [0.000001, 0.00001, 0.0001, 0.00005, 0.005, 0.003, 0.001, 0.0005, 0.1]
ALPHA = [0.003]

# ALL               = [3, 4, 5, 8]
FOURIER_BASIS_ORDER = [3]

# ALL    = [100, 200]
EPISODES = [200]

SKIP_EXISTING_PATH = False


def execute():
    for epsilon, epsilon_decay, alpha, fourier_basis_order, episodes in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY,
                    ALPHA, FOURIER_BASIS_ORDER, EPISODES):
        fourier_basis_order = int(fourier_basis_order)
        episodes = int(episodes)

        trials_dir = '{}__q_learning__e={}__d={}__a={}__f={}__ep={}' \
            .format(cartpole.ENV, epsilon, epsilon_decay,
                alpha, fourier_basis_order, episodes)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            agent = CartPole(fourier_basis_order=fourier_basis_order)
            episode_results = q_learning(agent=agent,
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes,
                    trials_dir=trials_dir)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
