import numpy as np

# Class describing a DeterministicPolicy.
# You should only interact with it by using the get_action()
# and set_action() functions.
# For example:
#
# policy = DeterministicPolicy(2, 2)
# for state in [StudentMDP.STATE_HAPPY, StudentMDP.STATE_SAD]:
#    policy.set_action(state, StudentMDP.ACTION_DRINK)


class DeterministicPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.state_action_map = np.zeros(num_states, dtype=np.int)

    def get_action(self, state):
        if (state < 0 or state >= self.num_states):
            raise Exception("Invalid state")
        return self.state_action_map[state]

    def set_action(self, state, action):
        if (state < 0 or state >= self.num_states):
            raise Exception("Invalid state")
        if (action < 0 or action >= self.num_actions):
            raise Exception("Invalid action")
        self.state_action_map[state] = action

def __init__(self):
    pass
