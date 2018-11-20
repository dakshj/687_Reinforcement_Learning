import numpy as np

from agent.agent_1 import Agent
from agent.non_tabular_agent import NonTabularAgent
from agent.tabular_agent import TabularAgent

EPISODES = 100


def sarsa(agent: Agent, alpha: float) -> list:
    # List of rewards across all episodes, for this one trial
    all_returns = []

    # Tabular Variables
    q = None

    # Non-Tabular Variables
    weights = None
    phi = None
    phi_next = None

    if isinstance(agent, TabularAgent):
        q = agent.init_q()
    elif isinstance(agent, NonTabularAgent):
        weights = agent.init_weights()
        phi = agent.get_phi()

    for episode in range(EPISODES):
        agent.reset_for_new_episode()

        state = agent.state

        action = agent.get_action(
                q_or_weights=q if isinstance(agent, TabularAgent) else weights
        )

        while not agent.has_terminated():
            reward, state_next = agent.take_action(action)

            action_next = agent.get_action(
                    q_or_weights=q if isinstance(agent, TabularAgent) else weights
            )

            if isinstance(agent, TabularAgent):
                q[(state, action)] += alpha * \
                                      (reward + agent.gamma *
                                       q[(state_next, action_next)]) - \
                                      q[(state, action)]
            elif isinstance(agent, NonTabularAgent):
                phi_next = agent.get_phi()

                action_index = agent.get_action_index(action)
                action_next_index = agent.get_action_index(action_next)

                q_w = np.dot(weights[action_index], phi)
                q_w_next = np.dot(weights[action_next_index], phi_next)

                weights[action_index] += \
                    alpha * (reward + agent.gamma * q_w_next - q_w) * phi

            state = state_next
            action = action_next
            phi = phi_next

            all_returns.append(agent.returns)

            # Episode end

    return all_returns
