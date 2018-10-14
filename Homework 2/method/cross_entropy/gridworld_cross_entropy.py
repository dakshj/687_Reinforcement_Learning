from method.cross_entropy import cross_entropy
from util.trial_plotting import save_trial

# Constants
ENV_GRIDWORLD = 'gridworld'
TRIALS_DIR = '{}_cross_entropy_trials'.format(ENV_GRIDWORLD)
TRIALS = 20
WHILE_LOOP_ITERATIONS_VALUES = [100]
K_VALUES = [300]
K_e_VALUES = [30]
N_VALUES = [50]


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        save_trial(
                            cross_entropy.cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial,
                                TRIALS, ENV_GRIDWORLD),
                            TRIALS_DIR
                        )


if __name__ == '__main__':
    execute()
