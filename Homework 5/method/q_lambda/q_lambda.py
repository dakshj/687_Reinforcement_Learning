import time
from copy import deepcopy

import numpy as np

from agent.agent import Agent, EPSILON_GREEDY
from agent.non_tabular.non_tabular_agent import NonTabularAgent
from agent.tabular.tabular_agent import TabularAgent


def q_lambda(agent: Agent, epsilon: float, epsilon_decay: float,
             alpha: float, trial: int, trials_total: int, episodes: int,
             trials_dir: str, lambda_: float,
             action_selection_method: str = EPSILON_GREEDY,
             sigma: float = None) -> list:
    # List of rewards across all episodes, for this one trial
    episode_returns = []

    weights = agent.init_weights()

    exec_time = time.time()
    for episode in range(episodes):
        print('Episode {} / {} in Trial {} / {} (Time = {} s, Time steps = {}, {})'
            .format(episode + 1, episodes, trial + 1, trials_total,
                round(time.time() - exec_time, 2), agent.time_step, trials_dir))
        exec_time = time.time()

        agent.reset_for_new_episode(epsilon=epsilon, sigma=sigma)

        state = agent.state

        e_trace = agent.init_e_trace()

        while not agent.has_terminated():
            action = agent.get_action(
                    weights=weights,
                    action_selection_method=action_selection_method
            )

            reward, state_next = agent.take_action(action)

            action_index = agent.get_action_index(action)

            max_q_value = agent.get_max_q_value(state=state_next, weights=weights)

            dq_dw = np.zeros_like(weights)
            delta = None

            # Calculate TD Error, and set dq_dw based on agent
            if isinstance(agent, TabularAgent):
                state_index = agent.get_state_index(state)

                delta = reward + \
                        agent.gamma * max_q_value - \
                        weights[state_index, action_index]

                dq_dw[state_index, action_index] = 1

            elif isinstance(agent, NonTabularAgent):
                q_w = agent.get_q_values_vector(
                        state=state, weights=weights)[action_index]

                delta = reward + agent.gamma * max_q_value - q_w

                dq_dw[action_index] = agent.get_phi(state)

                # If-Else end

            e_trace = e_trace * agent.gamma * lambda_ + dq_dw

            weights += alpha * delta * e_trace

            state = deepcopy(state_next)

            # Time step end

        episode_returns.append(agent.returns)

        epsilon *= epsilon_decay

        # Episode end

    return episode_returns
