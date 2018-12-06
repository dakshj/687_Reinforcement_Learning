import time
from copy import deepcopy

import numpy as np

from agent.agent import Agent, SOFTMAX
from agent.non_tabular.non_tabular_agent import NonTabularAgent
from agent.tabular.tabular_agent import TabularAgent


def actor_critic(agent: Agent, lambda_: float,
                 alpha_actor: float, alpha_critic: float, sigma: float,
                 trial: int, trials_total: int, episodes: int,
                 trials_dir: str) -> list:
    # List of rewards across all episodes, for this one trial
    episode_returns = []

    weights = agent.init_weights_actor_critic()

    theta = agent.init_weights()

    exec_time = time.time()
    for episode in range(episodes):
        print('Episode {} / {} in Trial {} / {} (Time = {} s, '
              'Time steps = {}, {})'
            .format(episode + 1, episodes, trial + 1, trials_total,
                round(time.time() - exec_time, 2), agent.time_step, trials_dir))
        exec_time = time.time()

        agent.reset_for_new_episode(sigma=sigma)

        e_v = agent.init_e_v()
        e_0 = agent.init_e_0()

        state = agent.state

        while not agent.has_terminated():
            action = agent.get_action(
                    weights=theta,
                    action_selection_method=SOFTMAX
            )

            action_index = agent.get_action_index(action)

            reward, state_next = agent.take_action(action)

            dv_dw = np.zeros_like(e_v)
            delta = None

            # Calculate TD Error, and set dv_dw & dlnpi_d0 based on agent
            if isinstance(agent, TabularAgent):
                state_index = agent.get_state_index(state)

                state_next_index = agent.get_state_index(state_next)

                delta = reward + \
                        agent.gamma * weights[state_next_index] \
                        - weights[state_index]

                dv_dw[state_index] = 1

            elif isinstance(agent, NonTabularAgent):
                # get_q_values_vector will now return a scalar
                #  because weights is a vector now, instead of a 2d array
                v_w = agent.get_q_values_vector(state=state, weights=weights)
                v_w_next = agent.get_q_values_vector(
                        state=state_next, weights=weights)

                delta = reward + agent.gamma * v_w_next - v_w

                dv_dw = agent.get_phi(state)

                # If-Else end

            # Critic Update
            e_v = e_v * agent.gamma * lambda_ + dv_dw
            weights += alpha_critic * delta * e_v

            # Actor Update
            # TODO Pass the appropriate parameters
            e_0 = e_0 * agent.gamma * lambda_ + get_dlnpi_d0()
            theta += alpha_actor * delta * e_0

            state = deepcopy(state_next)

            # Time step end

        episode_returns.append(agent.returns)

        # Episode end

    return episode_returns


def get_dlnpi_d0(agent, theta, state, action_index) -> np.ndarray:
    dlnpi_d0 = None

    pi = agent.get_q_values_vector(state=state, weights=theta)
    pi = np.exp(pi)
    pi /= np.sum(pi)

    if isinstance(agent, TabularAgent):
        # TODO Implement this
        pass

    elif isinstance(agent, NonTabularAgent):
        dlnpi_d0 = np.ones_like(theta) * agent.get_phi(state=state)

        for i in range(action_index):
            if i != action_index:
                dlnpi_d0[i] *= -pi[i]
            else:
                dlnpi_d0[i] *= 1 - pi[i]

    return dlnpi_d0
