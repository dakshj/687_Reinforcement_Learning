from agent import cartpole
from method.hill_climbing.hill_climbing import hill_climbing
from util.plot_trials import save_trial

# Constants
TRIALS_DIR = '{}_hill_climbing_trials'.format(cartpole.ENV)
TRIALS = 500
N_VALUES = [1]  # Set to 1 because cartpole is deterministic

# Hyper-parameters
WHILE_LOOP_ITERATIONS_VALUES = [300]
SIGMA_VALUES = [6]


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for sigma_hyp in SIGMA_VALUES:
                for N_hyp in N_VALUES:
                    save_trial(
                        hill_climbing(while_hyp, sigma_hyp, N_hyp, trial, TRIALS, cartpole.ENV),
                        TRIALS_DIR
                    )


if __name__ == '__main__':
    execute()
