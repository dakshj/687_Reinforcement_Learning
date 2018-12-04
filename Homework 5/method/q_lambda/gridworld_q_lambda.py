import os

from agent.tabular import gridworld
from agent.tabular.gridworld import GridWorld
from method.q_lambda.q_lambda import q_lambda
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 15

# ALL   = [0.98, 0.95, 0.9, 0.8, 0.4, 0.3, 0.1]
EPSILON = [0.98, 0.95, 0.9, 0.8, 0.4, 0.3, 0.1]

# `1` means no decay
# ALL         = [1, 0.99, 0.98, 0.95]
EPSILON_DECAY = [1, 0.99, 0.98, 0.95]

# ALL = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.3, 0.75, 0.25, 0.2]
ALPHA = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.3, 0.75, 0.25, 0.2]

# ALL    = [150]
EPISODES = [150]

# ALL  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
LAMBDA = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

SKIP_EXISTING_PATH = True


def execute():
    for epsilon, epsilon_decay, alpha, episodes, lambda_ in \
            random_hyperparameter_search(EPSILON, EPSILON_DECAY,
                    ALPHA, EPISODES, LAMBDA):
        episodes = int(episodes)

        trials_dir = '{}__q_learning__e={}__d={}__a={}__ep={}__l={}' \
            .format(gridworld.ENV, epsilon, epsilon_decay, alpha, episodes, lambda_)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        for trial in range(TRIALS):
            episode_results = q_lambda(agent=GridWorld(),
                    epsilon=epsilon, epsilon_decay=epsilon_decay,
                    alpha=alpha, trial=trial, trials_total=TRIALS, episodes=episodes,
                    trials_dir=trials_dir, lambda_=lambda_)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
