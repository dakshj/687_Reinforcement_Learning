import os

from agent.non_tabular import mountaincar
from agent.non_tabular.mountaincar import MountainCar
from method.reinforce.reinforce_with_baseline import reinforce_with_baseline
from method.reinforce.reinforce_without_baseline import reinforce_without_baseline
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL               = [3, 4, 5, 7, 8]
FOURIER_BASIS_ORDER = [4]

# ALL  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]
LAMBDA = [0.05]

# ALL       = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_THETA = [.005]

# ALL         = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_WEIGHTS = [0.005]

# ALL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SIGMA = [0.1]

# ALL    = [True, False]
WITH_BASELINE = [False]

EPISODES = 200

SKIP_EXISTING_PATH = False


def execute():
    for fourier_basis_order, lambda_, alpha_theta, alpha_weights, sigma, with_baseline in \
            random_hyperparameter_search(FOURIER_BASIS_ORDER, LAMBDA,
                    ALPHA_THETA, ALPHA_WEIGHTS, SIGMA, WITH_BASELINE):

        fourier_basis_order = int(fourier_basis_order)
        with_baseline = bool(with_baseline)

        trials_dir = '{}__reinforce__f={}__l={}__at={}__aw={}__s={}__bl={}' \
            .format(mountaincar.ENV, fourier_basis_order, lambda_, alpha_theta, alpha_weights,
                sigma,
                with_baseline)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)

        for trial in range(TRIALS):
            agent = MountainCar(fourier_basis_order=fourier_basis_order)

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
