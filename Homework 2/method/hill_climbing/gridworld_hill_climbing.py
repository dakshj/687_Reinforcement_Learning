from agent import gridworld
from method.hill_climbing.hill_climbing import hill_climbing
from util.plot.plot_trials import save_trial

# Constants
TRIALS_DIR = '{}_hill_climbing_trials'.format(gridworld.ENV)
TRIALS = 10000

# Hyper-parameters
# REMEMBER TO SET ASIDE PREVIOUS TRIALS DIR!
WHILE_LOOP_ITERATIONS_VALUES = [300]
SIGMA_VALUES = [2]
N_VALUES = [90]


def execute():
    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for sigma_hyp in SIGMA_VALUES:
                for N_hyp in N_VALUES:
                    save_trial(
                        hill_climbing(while_hyp, sigma_hyp, N_hyp, trial, TRIALS, gridworld.ENV),
                        TRIALS_DIR
                    )


if __name__ == '__main__':
    execute()
