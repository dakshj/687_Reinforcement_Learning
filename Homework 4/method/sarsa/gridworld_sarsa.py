from agent import gridworld
from agent.gridworld import GridWorld
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

EPSILON = [0.3]
ALPHA = [0.1]
EPISODES = [100, 200]


def execute():
    for epsilon, alpha, episodes in \
            random_hyperparameter_search(EPSILON, ALPHA, EPISODES):
        trials_dir = '{}__sarsa__e={}__a={}__ep={}' \
            .format(gridworld.ENV, epsilon, alpha, episodes)

        for trial in range(TRIALS):
            agent = GridWorld(epsilon=epsilon)
            episode_results = sarsa(agent=agent, alpha=alpha,
                    trial=trial, trials_total=TRIALS, episodes=episodes)

            save_trial(arr=episode_results, trials_dir=trials_dir)


if __name__ == '__main__':
    execute()
