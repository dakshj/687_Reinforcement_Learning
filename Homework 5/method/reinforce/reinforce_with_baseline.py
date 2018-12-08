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

            episode_vars_list.append(
                    EpisodeVars(state=state, state_next=state_next,
                            action=action, reward=reward)
            )

            # Episode end

        g_array = get_g_array(episode_vars_list=episode_vars_list,
                gamma=agent.gamma)

        delta_j_pi = np.zeros_like(theta)

        e_trace = agent.init_e_v()

        for t in range(len(episode_vars_list)):
            v_w, v_w_next, dv_dw = get__v_w__dv_dw(agent=agent, t=t,
                    episode_vars_list=episode_vars_list, weights=weights,
                    e_trace=e_trace)

            delta_j_pi += math.pow(agent.gamma, t) * \
                          (g_array[t] - v_w) * \
                          get_d_ln_pi__d_theta(agent=agent, theta=theta,
                                  state=episode_vars_list[t].state,
                                  action_index=agent.get_action_index(
                                          episode_vars_list[t].action)
                          )

            e_trace *= agent.gamma * lambda_ + dv_dw

            delta = episode_vars_list[t].reward + \
                    agent.gamma * v_w_next - v_w

            weights += alpha_weights * delta * e_trace

            # Time steps end

        theta += alpha_theta * delta_j_pi

        episode_returns.append(agent.returns)

        # Episode end

    return episode_returns


def get__v_w__dv_dw(agent: Agent, t: int, episode_vars_list: List[EpisodeVars],
                    weights: np.ndarray, e_trace: np.ndarray) -> tuple:
    dv_dw = np.zeros_like(e_trace)

    if isinstance(agent, TabularAgent):
        state_index = agent.get_state_index(episode_vars_list[t].state)
        state_next_index = agent.get_state_index(episode_vars_list[t].state_next)

        dv_dw[state_index] = 1

        return weights[state_index], weights[state_next_index], dv_dw

    elif isinstance(agent, NonTabularAgent):
        v_w = agent.get_q_values_vector(state=episode_vars_list[t].state,
                weights=weights)
        v_w_next = agent.get_q_values_vector(
                state=episode_vars_list[t].state_next,
                weights=weights)

        dv_dw = agent.get_phi(episode_vars_list[t].state)

        return v_w, v_w_next, dv_dw


def get_g_array(episode_vars_list: List[EpisodeVars], gamma: float) \
        -> np.ndarray:
    g_array = np.zeros(len(episode_vars_list))

    for i in range(len(g_array))[::-1]:
        g_array[i] = episode_vars_list[i].reward

        if i != len(g_array) - 1:
            g_array[i] += gamma * g_array[i + 1]

    return g_array
