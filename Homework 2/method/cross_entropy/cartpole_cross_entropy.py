from agent import cartpole
from method.cross_entropy.cross_entropy import cross_entropy
from util.plot.plot_trials import save_trial

# Constants
TRIALS_DIR = '{}_cross_entropy_trials'.format(cartpole.ENV)
TRIALS = 20
N_VALUES = [1]  # Set to 1 because cartpole is deterministic

# Hyper-parameters
WHILE_LOOP_ITERATIONS_VALUES = [25]
K_VALUES = [100]
K_e_VALUES = [50]


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        save_trial(
                            cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial,
                                TRIALS, cartpole.ENV),
                            TRIALS_DIR
                        )


if __name__ == '__main__':
    execute()
