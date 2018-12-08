import math
import time
from typing import List

import numpy as np

from agent.agent import Agent, SOFTMAX
from method.actor_critic.actor_critic import get_d_ln_pi__d_theta
from method.reinforce.model.episode_vars import EpisodeVars
from method.reinforce.reinforce_with_baseline import get_g_array


def reinforce_without_baseline(agent: Agent, alpha_theta: float,
                               sigma: float, trial: int, trials_total: int,
                               episodes: int, trials_dir: str) -> list:
    # List of rewards across all episodes, for this one trial
    episode_returns = []

    theta = agent.init_weights()

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

        for t in range(len(episode_vars_list)):
            delta_j_pi += math.pow(agent.gamma, t) * \
                          g_array[t] * \
                          get_d_ln_pi__d_theta(agent=agent, theta=theta,
                                  state=episode_vars_list[t].state,
                                  action_index=agent.get_action_index(
                                          episode_vars_list[t].action)
                          )

            # Time steps end

        theta += alpha_theta * delta_j_pi

        episode_returns.append(agent.returns)

        # Episode end

    return episode_returns
