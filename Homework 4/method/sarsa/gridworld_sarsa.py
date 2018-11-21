import os

from agent import gridworld
from agent.gridworld import GridWorld
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL   = [0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
EPSILON = [0.5]

# `1` means no decay
# ALL         = [1, 0.98, 0.95, 0.9, 0.88, 0.85]
EPSILON_DECAY = [0.9]

# ALL = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.3, 0.75, 0.25, 0.2]
ALPHA = [0.3]

# ALL    = [100, 200, 300, 500, 700, 1000]
EPISODES = [300]


def execute():
    for epsilon, epsilon_decay, alpha, episodes in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY, ALPHA, EPISODES):
        episodes = int(episodes)

        trials_dir = '{}__sarsa__e={}__d={}__a={}__ep={}' \
            .format(gridworld.ENV, epsilon, epsilon_decay, alpha, episodes)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        skip_existing_path = True
        if skip_existing_path and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            episode_results = sarsa(agent=GridWorld(),
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
