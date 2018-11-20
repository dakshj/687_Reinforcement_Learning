from agent import gridworld
from agent.gridworld import GridWorld
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

EPSILON = [0.3]
ALPHA = [0.1]


def execute():
    for epsilon, alpha in random_hyperparameter_search(EPSILON, ALPHA):
        trials_dir = '{}__sarsa__e={}__a={}' \
            .format(gridworld.ENV, epsilon, alpha)

        for trial in range(TRIALS):
            agent = GridWorld(epsilon=epsilon)
            episode_results = sarsa(agent=agent, alpha=alpha,
                    trial=trial, trials_total=TRIALS)

            save_trial(arr=episode_results, trial_dir=trials_dir)


if __name__ == '__main__':
    execute()
