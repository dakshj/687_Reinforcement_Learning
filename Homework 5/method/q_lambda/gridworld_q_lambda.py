import os

from agent.tabular import gridworld
from agent.tabular.gridworld import GridWorld
from method.q_lambda.q_lambda import q_lambda
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL   = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
EPSILON = [0.35]

# `1` means no decay
# ALL         = [1, 0.95, 0.9, 0.8, 0.5]
EPSILON_DECAY = [0.5]

# ALL = [0.1, 0.05, 0.01, 0.005, 0.001, 0.2]
ALPHA = [0.1]

# ALL    = [100, 200, 300, 400]
EPISODES = [200]

# ALL  = []
LAMBDA = []

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
