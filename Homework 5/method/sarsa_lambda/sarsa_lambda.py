import time
from copy import deepcopy

import numpy as np

from agent.agent import Agent, EPSILON_GREEDY
from agent.non_tabular.non_tabular_agent import NonTabularAgent
from agent.tabular.tabular_agent import TabularAgent


def sarsa_lambda(agent: Agent, epsilon: float, epsilon_decay: float,
                 alpha: float, trial: int, trials_total: int, episodes: int,
                 trials_dir: str, lambda_: float,
                 action_selection_method: str = EPSILON_GREEDY,
                 sigma: float = None) -> list:
    # List of rewards across all episodes, for this one trial
    episode_returns = []

    # Tabular Variables
    q = None

    # Non-Tabular Variables
    weights = None

    if isinstance(agent, TabularAgent):
        q = agent.init_q()
    elif isinstance(agent, NonTabularAgent):
        weights = agent.init_weights()

    exec_time = time.time()
    for episode in range(episodes):
        print('Episode {} / {} in Trial {} / {} (Time = {} s, '
              'Time steps = {}, {})'
            .format(episode + 1, episodes, trial + 1, trials_total,
                round(time.time() - exec_time, 2), agent.time_step, trials_dir))
        exec_time = time.time()

        agent.reset_for_new_episode(epsilon=epsilon, sigma=sigma)

        state = agent.state

        action = agent.get_action(
                q_or_weights=q if isinstance(agent, TabularAgent) else weights,
                action_selection_method=action_selection_method
        )

        e_trace = agent.init_e_trace()

        while not agent.has_terminated():
            reward, state_next = agent.take_action(action)

            action_next = agent.get_action(
                    q_or_weights=q if isinstance(agent, TabularAgent) else weights,
                    action_selection_method=action_selection_method
            )

            action_index = agent.get_action_index(action)
            action_next_index = agent.get_action_index(action_next)

            dq_dw = None
            if isinstance(agent, TabularAgent):
                dq_dw = 1
            elif isinstance(agent, NonTabularAgent):
                dq_dw = np.zeros_like(weights)
                dq_dw[action_index] = agent.get_phi(state)
            e_trace = agent.gamma * lambda_ * e_trace + dq_dw

            if isinstance(agent, TabularAgent):
                state_index = agent.get_state_index(state)

                state_next_index = agent.get_state_index(state_next)

                delta = reward + \
                        agent.gamma * q[state_next_index, action_next_index] \
                        - q[state_index, action_index]

                q += alpha * delta * e_trace

            elif isinstance(agent, NonTabularAgent):
                q_w = agent.get_q_values_vector(
                        state=state, q_or_weights=weights)[action_index]
                q_w_next = agent.get_q_values_vector(
                        state=state_next, q_or_weights=weights)[action_next_index]

                delta = reward + agent.gamma * q_w_next - q_w

                weights += alpha * delta * e_trace

                # If-Else end

            state = deepcopy(state_next)
            action = deepcopy(action_next)

            # Time step end

        episode_returns.append(agent.returns)

        epsilon *= epsilon_decay

        # Episode end

    return episode_returns
