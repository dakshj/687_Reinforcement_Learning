import os

from agent.tabular import gridworld
from agent.tabular.gridworld import GridWorld
from method.actor_critic.actor_critic import actor_critic
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

# ALL  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]
LAMBDA = [0.9]

# ALL       = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_ACTOR = [0.1]

# ALL        = [0.1, 0.01, 0.001, 0.005, 0.05]
ALPHA_CRITIC = [0.1]

# ALL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SIGMA = [0.9]

EPISODES = 150

SKIP_EXISTING_PATH = False


def execute():
    for lambda_, alpha_actor, alpha_critic, sigma in \
            random_hyperparameter_search(LAMBDA, ALPHA_ACTOR, ALPHA_CRITIC, SIGMA):

        trials_dir = '{}__actor_critic__l={}__aa={}__ac={}__s={}' \
            .format(gridworld.ENV, lambda_, alpha_actor, alpha_critic, sigma)

        # Skipping existing dirs helps in parallelization by skipping
        # those hyperparams that have already been checked
        if SKIP_EXISTING_PATH and os.path.exists(trials_dir):
            continue

        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)

        for trial in range(TRIALS):
            agent = GridWorld()
            episode_results = actor_critic(agent=agent, lambda_=lambda_,
                    alpha_actor=alpha_actor, alpha_critic=alpha_critic, sigma=sigma,
                    trial=trial, trials_total=TRIALS, episodes=EPISODES,
                    trials_dir=trials_dir)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
