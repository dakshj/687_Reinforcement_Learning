import os

from agent.non_tabular import cartpole
from agent.non_tabular.cartpole import CartPole
from agent.non_tabular.non_tabular_agent import NonTabularAgent
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 20

# ALL   = [0.9, 0.95, 0.7, 0.75, 0.8]
EPSILON = [0.95]

# `1` means no decay
# ALL         = [1, 0.99, 0.98, 0.9]
EPSILON_DECAY = [1]

# ALL    = [150]
EPISODES = [150]

# ALL   = [3, 5, 8]
TILINGS = [8]

# ALL            = [10, 15, 18, 20]
TILES_PER_TILING = [20]

SKIP_EXISTING_PATH = False


def execute():
    for epsilon, epsilon_decay, \
        episodes, tilings, tiles_per_tiling in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY,
                    EPISODES, TILINGS, TILES_PER_TILING):
        episodes = int(episodes)
        tilings = int(tilings)
        tiles_per_tiling = int(tiles_per_tiling)

        trials_dir = '{}__sarsa_tc__e={}__d={}__ep={}__t1={}__t2={}' \
            .format(cartpole.ENV, epsilon, epsilon_decay,
                episodes, tilings, tiles_per_tiling)

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
                    alpha=.1 / tilings, trial=trial, trials_total=TRIALS,
                    episodes=episodes, trials_dir=trials_dir)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
