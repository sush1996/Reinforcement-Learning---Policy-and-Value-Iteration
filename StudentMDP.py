# Basic Class that implements the StudentMDP
# Don't touch any internal data.
# Use get_data() to get the transition probabilities and rewards
class StudentMDP(object):
    ACTION_DRINK = 0
    ACTION_STUDY = 1
    STATE_HAPPY = 0
    STATE_SAD = 1

    def __init__(self):
        self.data = [[[[], []], [[], []]], [[[], []], [[], []]]]
        self.data[StudentMDP.STATE_HAPPY][StudentMDP.ACTION_DRINK][StudentMDP.STATE_HAPPY] = (1. , 10.)
        self.data[StudentMDP.STATE_HAPPY][StudentMDP.ACTION_DRINK][StudentMDP.STATE_SAD] = (0., 0.)
        self.data[StudentMDP.STATE_HAPPY][StudentMDP.ACTION_STUDY][StudentMDP.STATE_HAPPY] = (0.2, -10.)
        self.data[StudentMDP.STATE_HAPPY][StudentMDP.ACTION_STUDY][StudentMDP.STATE_SAD] = (0.8, -10.)
        self.data[StudentMDP.STATE_SAD][StudentMDP.ACTION_DRINK][StudentMDP.STATE_HAPPY] = (0.8, 40.)
        self.data[StudentMDP.STATE_SAD][StudentMDP.ACTION_DRINK][StudentMDP.STATE_SAD] = (0.2, 40)
        self.data[StudentMDP.STATE_SAD][StudentMDP.ACTION_STUDY][StudentMDP.STATE_HAPPY] = (0.8, 20)
        self.data[StudentMDP.STATE_SAD][StudentMDP.ACTION_STUDY][StudentMDP.STATE_SAD] = (0.2, 20)

    # For a given state (i.e. StudentMDP.STATE_HAPPY), 
    # state_prime, and action (i.e. ACTION_DRINK), this function will return a tuple of
    # (transition probability, reward).
    # transition probability is P(state_prime | state, action)
    # and reward is just R(state, action)
    def get_data(self, state, action, state_prime):
        if (state < 0 or state >= self.get_num_states()):
            raise Exception("Invalid state")
        if (state_prime < 0 or state_prime >= self.get_num_states()):
            raise Exception("Invalid state_prime")
        if (action < 0 or action >= self.get_num_actions()):
            raise Exception("Invalid action")
        return self.data[state][action][state_prime]

    # Gets the number of states
    def get_num_states(self):
        return 2

    # Gets the number of actions
    def get_num_actions(self):
        return 2
