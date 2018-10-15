from agent import gridworld
from method.cross_entropy.cross_entropy import cross_entropy
from util.plot.plot_trials import save_trial

# Constants
TRIALS_DIR = '{}_cross_entropy_trials'.format(gridworld.ENV)
TRIALS = 10000

# Hyper-parameters
# REMEMBER TO SET ASIDE PREVIOUS TRIALS DIR!
WHILE_LOOP_ITERATIONS_VALUES = [50]
K_VALUES = [800]
K_e_VALUES = [200]
N_VALUES = [5]
EPSILON_VALUES = [0.0001]
SIGMA_MULTIPLIER_VALUES = [150]


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        for epsilon_hyp in EPSILON_VALUES:
                            for sigma_multiplier_hyp in SIGMA_MULTIPLIER_VALUES:
                                save_trial(
                                    cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial,
                                        TRIALS, gridworld.ENV, epsilon_hyp, sigma_multiplier_hyp),
                                    TRIALS_DIR
                                )


if __name__ == '__main__':
    execute()
