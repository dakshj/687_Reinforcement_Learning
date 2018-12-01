import os

from agent.tabular import gridworld
from agent.tabular.gridworld import GridWorld
from method.sarsa_lambda.sarsa_lambda import sarsa_lambda
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 30

# ALL   = [0.02, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
EPSILON = [0.5]

# `1` means no decay
# ALL         = [1, 0.98, 0.95, 0.9, 0.88, 0.85]
EPSILON_DECAY = [0.9]

# ALL = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.3, 0.75, 0.25, 0.2]
ALPHA = [0.3]

# ALL    = [100, 200, 300, 500, 700, 1000]
EPISODES = [300]

# ALL  = [0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
LAMBDA = [0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

SKIP_EXISTING_PATH = True


def execute():
    for epsilon, epsilon_decay, alpha, episodes, lambda_ in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY, ALPHA, EPISODES,
                    LAMBDA):
        episodes = int(episodes)

        trials_dir = '{}__sarsa_lambda__e={}__d={}__a={}__ep={}__l={}' \
            .format(gridworld.ENV, epsilon, epsilon_decay, alpha, episodes, lambda_)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            episode_results = sarsa_lambda(agent=GridWorld(),
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes,
                    trials_dir=trials_dir, lambda_=lambda_)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
