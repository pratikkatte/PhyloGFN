class Trajectory(object):

    def __init__(self, initial_state):
        self.current_state = initial_state
        self.transitions = []
        self.reward = None
        self.done = False
        self.actions = []

    def update(self, next_state, action, reward, done):
        self.transitions.append(
            [self.current_state, next_state, action, reward, done]
        )
        self.current_state = next_state
        self.actions.append(action)
        self.done = done
        self.reward = reward

    def update_reward(self, reward):
        self.transitions[-1][-2] = reward
        self.reward = reward