import math
import time
from typing import List

import numpy as np

from agent.agent import Agent, SOFTMAX
from agent.non_tabular.non_tabular_agent import NonTabularAgent
from agent.tabular.tabular_agent import TabularAgent
from method.actor_critic.actor_critic import get_d_ln_pi__d_theta
from method.reinforce.model.episode_vars import EpisodeVars


def reinforce_with_baseline(agent: Agent, lambda_: float,
                            alpha_weights: float, alpha_theta: float,
                            sigma: float, trial: int, trials_total: int,
                            episodes: int, trials_dir: str) -> list:
    # List of rewards across all episodes, for this one trial
    episode_returns = []

    theta = agent.init_weights()

    weights = agent.init_weights_actor_critic()

    exec_time = time.time()
    for episode in range(episodes):
        print('Episode {} / {} in Trial {} / {} (Time = {} s, '
              'Time steps = {}, {})'
            .format(episode + 1, episodes, trial + 1, trials_total,
                round(time.time() - exec_time, 2), agent.time_step, trials_dir))
        exec_time = time.time()

        agent.reset_for_new_episode(sigma=sigma)

        episode_vars_list: List[EpisodeVars] = []

        while not agent.has_terminated():
            state = agent.state

            action = agent.get_action(weights=theta,
                    action_selection_method=SOFTMAX)

            reward, state_next = agent.take_action(action)

            current_returns = agent.returns

            episode_vars_list.append(
                    EpisodeVars(state=state, state_next=state_next,
                            action=action, reward=reward,
                            current_returns=current_returns)
            )

            # Episode end

        delta_j_pi = np.zeros_like(theta)

        e_trace = agent.init_e_v()

        for t in range(len(episode_vars_list)):
            v_w, v_w_next = get__v_w__values(agent=agent, t=t,
                    episode_vars_list=episode_vars_list, weights=weights)

            delta_j_pi += math.pow(agent.gamma, t) * \
                          (episode_vars_list[t].current_returns - v_w) * \
                          get_d_ln_pi__d_theta(agent=agent, theta=theta,
                                  state=episode_vars_list[t].state,
                                  action_index=agent.get_action_index(
                                          episode_vars_list[t].action)
                          )

            e_trace *= agent.gamma * lambda_

            delta = episode_vars_list[t].reward + \
                    agent.gamma * v_w_next - v_w

            weights += alpha_weights * delta * e_trace

            # Time steps end

        theta += alpha_theta * delta_j_pi

        episode_returns.append(agent.returns)

        # Episode end

    return episode_returns


def get__v_w__values(agent: Agent, t: int, episode_vars_list: List[EpisodeVars],
                     weights: np.ndarray) -> tuple:
    if isinstance(agent, TabularAgent):
        state_index = agent.get_state_index(episode_vars_list[t].state)
        state_next_index = agent.get_state_index(episode_vars_list[t].state_next)

        return weights[state_index], weights[state_next_index]

    elif isinstance(agent, NonTabularAgent):
        v_w = agent.get_q_values_vector(state=episode_vars_list[t].state,
                weights=weights)
        v_w_next = agent.get_q_values_vector(
                state=episode_vars_list[t].state_next,
                weights=weights)

        return v_w, v_w_next
