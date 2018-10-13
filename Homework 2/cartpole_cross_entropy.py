import numpy as np

import cross_entropy

# Constants
ENV_CARTPOLE = 'cartpole'
TRIALS_DIR = '{}_cross_entropy_trials'.format(ENV_CARTPOLE)
TRIALS = 1
WHILE_LOOP_ITERATIONS_VALUES = [5]
K_VALUES = [10]
K_e_VALUES = [2]
N_VALUES = [1]  # Set to 1 because cartpole is deterministic


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        cross_entropy.save_trial(
                            cross_entropy.cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial,
                                TRIALS, ENV_CARTPOLE),
                            TRIALS_DIR
                        )


def generate_initial_cartpole_policy():
    return np.random.uniform(-10, 10, (4,))


if __name__ == '__main__':
    execute()
