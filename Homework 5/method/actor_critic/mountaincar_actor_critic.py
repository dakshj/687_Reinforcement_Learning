import os

from agent.non_tabular import mountaincar
from agent.non_tabular.mountaincar import MountainCar
from method.actor_critic.actor_critic import actor_critic
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL               = [3, 4, 5, 7, 8]
FOURIER_BASIS_ORDER = [7]

# ALL  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
LAMBDA = [0.3]

# ALL       = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_ACTOR = [0.001]

# ALL        = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_CRITIC = [0.05]

# ALL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIGMA = [0.9]

EPISODES = 150

SKIP_EXISTING_PATH = False


def execute():
    for fourier_basis_order, lambda_, alpha_actor, alpha_critic, sigma in \
            random_hyperparameter_search(FOURIER_BASIS_ORDER, LAMBDA,
                    ALPHA_ACTOR, ALPHA_CRITIC, SIGMA):

        fourier_basis_order = int(fourier_basis_order)

        trials_dir = '{}__actor_critic__f={}__l={}__aa={}__ac={}__s={}' \
            .format(mountaincar.ENV, fourier_basis_order, lambda_,
                alpha_actor, alpha_critic, sigma)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)

        for trial in range(TRIALS):
            agent = MountainCar(fourier_basis_order=fourier_basis_order)
            episode_results = actor_critic(agent=agent, lambda_=lambda_,
                    alpha_actor=alpha_actor, alpha_critic=alpha_critic, sigma=sigma,
                    trial=trial, trials_total=TRIALS, episodes=EPISODES,
                    trials_dir=trials_dir)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
