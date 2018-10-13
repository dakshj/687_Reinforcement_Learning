import numpy as np

from cross_entropy import ENV_CARTPOLE
from cross_entropy import cross_entropy
from cross_entropy import save_trial

# Constants
TRIALS_DIR = '{}_cross_entropy_trials'.format(ENV_CARTPOLE)
TRIALS = 20
WHILE_LOOP_ITERATIONS_VALUES = [100]
K_VALUES = [300]
K_e_VALUES = [30]
N_VALUES = [1]  # Set to 1 because cartpole is deterministic


def execute_cartpole():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        save_trial(
                            cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial, TRIALS,
                                ENV_CARTPOLE),
                            TRIALS_DIR
                        )


def generate_initial_cartpole_policy():
    return np.random.uniform(-10, 10, (4,))


if __name__ == '__main__':
    execute_cartpole()
