import time

import numpy as np

from agent.agent import Agent
from agent.non_tabular_agent import NonTabularAgent
from agent.tabular_agent import TabularAgent


def sarsa(agent: Agent, epsilon: float, epsilon_decay: float,
          alpha: float, trial: int, trials_total: int, episodes: int) -> list:
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
        print('Episode {} / {} in Trial {} / {} (Time = {} s, Time steps = {})'
            .format(episode + 1, episodes, trial + 1, trials_total,
                round(time.time() - exec_time, 2), agent.time_step))
        exec_time = time.time()

        agent.reset_for_new_episode(epsilon=epsilon)

        state = agent.state
        phi = agent.get_phi()

        action = agent.get_action(
                q_or_weights=q if isinstance(agent, TabularAgent) else weights
        )

        while not agent.has_terminated():
            reward, state_next = agent.take_action(action)

            action_next = agent.get_action(
                    q_or_weights=q if isinstance(agent, TabularAgent) else weights
            )

            action_index = agent.get_action_index(action)
            action_next_index = agent.get_action_index(action_next)

            phi_next = None

            if isinstance(agent, TabularAgent):
                state_index = agent.get_state_index(state)

                state_next_index = agent.get_state_index(state_next)

                q[state_index, action_index] += \
                    alpha * (reward + agent.gamma *
                             q[state_next_index, action_next_index]) - \
                    q[state_index, action_index]
            elif isinstance(agent, NonTabularAgent):
                phi_next = agent.get_phi()

                q_w = np.dot(weights[action_index], phi)
                q_w_next = np.dot(weights[action_next_index], phi_next)

                weights[action_index] += \
                    alpha * (reward + agent.gamma * q_w_next - q_w) * phi

                # If-Else end

            state = state_next
            action = action_next
            phi = phi_next

            # Time step end

        episode_returns.append(agent.returns)

        epsilon *= epsilon_decay

        # Episode end

    return episode_returns
