import os

from agent.tabular import gridworld
from agent.tabular.gridworld import GridWorld
from method.reinforce.reinforce_with_baseline import reinforce_with_baseline
from method.reinforce.reinforce_without_baseline import reinforce_without_baseline
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]
LAMBDA = [0.05]

# ALL       = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_THETA = [0.1]

# ALL         = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_WEIGHTS = [0.1]

# ALL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SIGMA = [1]

# ALL    = [True, False]
WITH_BASELINE = [True]

EPISODES = 800

SKIP_EXISTING_PATH = False


def execute():
    for lambda_, alpha_theta, alpha_weights, sigma, with_baseline in \
            random_hyperparameter_search(LAMBDA,
                    ALPHA_THETA, ALPHA_WEIGHTS, SIGMA, WITH_BASELINE):

        with_baseline = bool(with_baseline)

        trials_dir = '{}__reinforce__l={}__at={}__aw={}__s={}__bl={}' \
            .format(gridworld.ENV, lambda_, alpha_theta, alpha_weights, sigma,
                with_baseline)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)

        for trial in range(TRIALS):
            agent = GridWorld()

            if with_baseline:
                episode_results = \
                    reinforce_with_baseline(agent=agent, lambda_=lambda_,
                            alpha_theta=alpha_theta, alpha_weights=alpha_weights,
                            sigma=sigma, trial=trial, trials_total=TRIALS,
                            episodes=EPISODES, trials_dir=trials_dir)
            else:
                episode_results = \
                    reinforce_without_baseline(agent=agent,
                            alpha_theta=alpha_theta,
                            sigma=sigma, trial=trial, trials_total=TRIALS,
                            episodes=EPISODES, trials_dir=trials_dir)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
