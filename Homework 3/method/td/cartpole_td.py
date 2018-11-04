# TODO Change this file to be TD!

from agent import cartpole
from method.td.td import td
from util.plot.plot_trials import save_trial

# Constants
TRIALS_DIR = '{}_td_trials'.format(cartpole.ENV)
TRIALS = 10000
N_VALUES = [1]  # Set to 1 because cartpole is deterministic

# Hyper-parameters
# REMEMBER TO SET ASIDE PREVIOUS TRIALS DIR!
WHILE_LOOP_ITERATIONS_VALUES = [15]
K_VALUES = [80]
K_e_VALUES = [20]
EPSILON_VALUES = [0.0001]
SIGMA_MULTIPLIER_VALUES = [20]


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        for epsilon_hyp in EPSILON_VALUES:
                            for sigma_multiplier_hyp in SIGMA_MULTIPLIER_VALUES:
                                save_trial(
                                    td(while_hyp, K_hyp, K_e_hyp, N_hyp, trial,
                                        TRIALS, cartpole.ENV, epsilon_hyp, sigma_multiplier_hyp),
                                    TRIALS_DIR
                                )


if __name__ == '__main__':
    execute()
