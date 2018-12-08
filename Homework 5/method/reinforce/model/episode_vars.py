class EpisodeVars:
    def __init__(self, state, state_next, action, reward, current_returns):
        self.state = state
        self.state_next = state_next
        self.action = action
        self.reward = reward
        self.current_returns = current_returns
