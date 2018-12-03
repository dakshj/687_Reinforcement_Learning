import time

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
        print('Episode {} / {} in Trial {} / {} (Time = {} s, Time steps = {}, {})'
            .format(episode + 1, episodes, trial + 1, trials_total,
                round(time.time() - exec_time, 2), agent.time_step, trials_dir))
        exec_time = time.time()

        agent.reset_for_new_episode(epsilon=epsilon, sigma=sigma)

        state = agent.state

        e_trace = agent.init_e_trace()

        while not agent.has_terminated():
            action = agent.get_action(
                    q_or_weights=q if isinstance(agent, TabularAgent) else weights,
                    action_selection_method=action_selection_method
            )

            reward, state_next = agent.take_action(action)

            action_index = agent.get_action_index(action)

            del_by_del_part_of_e_trace = None
            if isinstance(agent, TabularAgent):
                del_by_del_part_of_e_trace = 1
            elif isinstance(agent, NonTabularAgent):
                del_by_del_part_of_e_trace = agent.get_phi(state)
            e_trace[action_index] = \
                agent.gamma * lambda_ * e_trace[action_index] \
                + del_by_del_part_of_e_trace

            max_q_value = agent.get_max_q_value(
                    q_or_weights=q if isinstance(agent, TabularAgent) else weights)

            if isinstance(agent, TabularAgent):
                state_index = agent.get_state_index(state)

                delta = reward + \
                        agent.gamma * max_q_value - q[state_index, action_index]

                q[state_index, action_index] += alpha * delta * \
                                                e_trace[state_index, action_index]

            elif isinstance(agent, NonTabularAgent):
                q_w = agent.get_q_values_vector(
                        state=state, q_or_weights=weights)[action_index]

                delta = reward + agent.gamma * max_q_value - q_w

                weights[action_index] += alpha * delta * e_trace[action_index]

                # If-Else end

            state = state_next

            # Time step end

        episode_returns.append(agent.returns)

        epsilon *= epsilon_decay

        # Episode end

    return episode_returns
