from agent import Agent

EPISODES = 100


def sarsa(agent: Agent, alpha: float) -> list:
    # List of rewards across all episodes, for this one trial
    all_returns = []

    # TODO What happens to q when we have a non-tabular policy?
    # TODO maybe use if-else bby checking for is_tabular to init q and w respectively
    q = agent.init_q()
    w = agent.init_w()

    for episode in range(EPISODES):
        agent.reset_for_new_episode()

        s = agent.get_state()

        a = agent.get_action(agent.get_policy_from_q(q))

        while not agent.has_terminated():
            r, s_next = agent.take_action(a)

            a_next = agent.get_action(agent.get_policy_from_q(q))

            if agent.is_tabular():
                q[(s, a)] += alpha * (r + agent.gamma() * q[(s_next, a_next)]) \
                             - q[(s, a)]
            else:
                # TODO
                del_part = None

                w += alpha * (r + agent.gamma() * q[s_next, a_next] - q[s, a]) * del_part

            s = s_next
            a = a_next

            all_returns.append(agent.returns)

    return all_returns
