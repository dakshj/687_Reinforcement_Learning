import os

from agent.non_tabular import cartpole
from agent.non_tabular.cartpole import CartPole
from agent.non_tabular.non_tabular_agent import NonTabularAgent
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL   = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
EPSILON = [0.35]

# `1` means no decay
# ALL         = [1, 0.98]
EPSILON_DECAY = [0.98]

# BAD = [0.1]
# ALL = [0.000001, 0.00001, 0.0001, 0.00005, 0.001, 0.0005, 0.1]
# TODO Alpha needs to be 1/num tilings
ALPHA = [0.001]

# ALL               = [3, 4, 5]
FOURIER_BASIS_ORDER = [5]

# ALL    = [100, 200]
EPISODES = [400]

# ALL   = []
TILINGS = []

# ALL            = []
TILES_PER_TILING = []

SKIP_EXISTING_PATH = True


def execute():
    for epsilon, epsilon_decay, alpha, \
        episodes, tilings, tiles_per_tiling in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY,
                    ALPHA, TILINGS, TILES_PER_TILING):
        episodes = int(episodes)
        tilings = int(tilings)
        tiles_per_tiling = int(tiles_per_tiling)

        trials_dir = '{}__sarsa__e={}__d={}__a={}__ep={}__t1={}__t2={}' \
            .format(cartpole.ENV, epsilon, epsilon_decay,
                alpha, episodes, tilings, tiles_per_tiling)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            agent = CartPole(
                    function_approximation_method=NonTabularAgent.TILE_CODING,
                    tilings=tilings, tiles_per_tiling=tiles_per_tiling
            )
            episode_results = sarsa(agent=agent,
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
