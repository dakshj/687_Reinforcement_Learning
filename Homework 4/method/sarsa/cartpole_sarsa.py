from agent import cartpole
from agent.cartpole import CartPole
from method.sarsa.sarsa import sarsa
from util.plot.plot_trials import save_trial
from util.random_hyperparameter_search import random_hyperparameter_search

TRIALS = 100

EPSILON = [0.3]
ALPHA = [0.1]
FOURIER_BASIS_ORDER = [3]


def execute():
    for epsilon, alpha, fourier_basis_order in \
            random_hyperparameter_search(EPSILON, ALPHA, FOURIER_BASIS_ORDER):
        fourier_basis_order = int(fourier_basis_order)

        trials_dir = '{}__sarsa__e={}__a={}__f={}' \
            .format(cartpole.ENV, epsilon, alpha, fourier_basis_order)

        for trial in range(TRIALS):
            agent = CartPole(epsilon=epsilon, fourier_basis_order=fourier_basis_order)
            episode_results = sarsa(agent=agent, alpha=alpha,
                    trial=trial, trials_total=TRIALS)

            save_trial(arr=episode_results, trial_dir=trials_dir)


if __name__ == '__main__':
    execute()
