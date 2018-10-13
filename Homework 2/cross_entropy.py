import time

import matplotlib.pyplot as plt
import numpy as np

import gridworld


def softmax(row):
    exp_row = np.exp(row)
    return exp_row / np.sum(exp_row, axis=0)


def convert_theta_to_table(theta):
    table = np.reshape(theta, (-1, 4))
    return np.apply_along_axis(softmax, axis=1, arr=table)


def generate_initial_tabular_softmax_policy():
    return np.random.uniform(0, 1, (92,))


TRIALS = 20
WHILE_LOOP_ITERATIONS_VALUES = [100]
K_VALUES = [300]
K_e_VALUES = [30]
N_VALUES = [50]
EPSILON = 0.0001


def execute():
    all_results = []

    for trial in range(TRIALS):
        for while_hyp in WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in K_VALUES:
                for K_e_hyp in K_e_VALUES:
                    for N_hyp in N_VALUES:
                        trial_results = cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial)
                        all_results.append(trial_results)

    # Save results to file
    np.save('all_results_{}'.format(time.time()), all_results)

    plot_results(all_results)


def plot_results(results):
    results = np.mean(results, axis=0)
    print(np.shape(results))
    plt.plot(results)
    plt.show()


def cross_entropy(while_limit, K, K_e, N, trial):
    theta = generate_initial_tabular_softmax_policy()
    sigma = np.identity(92)

    list_of__theta_k__vs__J_k_hat = []

    trial_results = []

    for while_i in range(while_limit):
        print('{} / {} in trial {} / {}'
            .format(while_i, while_limit, trial, TRIALS))

        for _ in range(K):
            theta_k = np.random.multivariate_normal(theta, sigma)

            table = convert_theta_to_table(theta_k)

            gridworld_results = gridworld.execute(episodes=N, policy_table=table)
            trial_results.extend(gridworld_results)

            J_k_hat = np.mean(gridworld_results)

            list_of__theta_k__vs__J_k_hat.append((theta_k, J_k_hat))

            # End K loop

        list_of__theta_k__vs__J_k_hat.sort(key=lambda x: x[1], reverse=True)

        filtered_theta_k_list = [x for (x, _) in list_of__theta_k__vs__J_k_hat][:K_e]

        theta_k_sum = np.sum(filtered_theta_k_list, axis=0)

        theta = 1 / K_e * theta_k_sum

        summation_part = [(x - theta) * np.transpose((x - theta)) for x in filtered_theta_k_list]
        summation_part = np.sum(summation_part, axis=0)

        sigma = (1 / (EPSILON + K_e)) * ((EPSILON * np.identity(92)) + summation_part)

        # End While loop

    return trial_results


if __name__ == '__main__':
    execute()
