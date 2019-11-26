import numpy as np
from itertools import permutations
import pandas as pd
import matplotlib.pyplot as plt

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


# TODO: Fill this function in
# This function implements Policy Evaluation. It takes in a StudentMDP 
# as well as a policy to evaluate and returns a numpy array of size (num_states).
# where the i'th index of the array is the value of the i'th state
def PolicyEvaluation(mdp, policy, gamma):
	
	states = [mdp.STATE_HAPPY, mdp.STATE_SAD]

	delta = [0.0,0.0]
	value = [0.0,0.0]
	v = [0.0,0.0]
	happy = []
	sad = []

	while True:
			
		delta = [0.0, 0.0]

		for i in range(mdp.get_num_states()): 
			v[i] = value[i]
			temp = 0.0
			for j in range(mdp.get_num_states()):
				temp = temp + mdp.get_data(states[i], policy.get_action(states[i]), states[j])[0]*mdp.get_data(states[i],
		 					policy.get_action(states[i]), states[j])[1] + mdp.get_data(states[i], policy.get_action(states[i]), states[j])[0]*gamma*value[j]
			value[i] = temp
			
			if i == 0:
				happy = happy + [value[i]]
			if i == 1:
				sad = sad + [value[i]] 
			
			delta[i] = max(delta[i], abs(v[i] - value[i]))

    		if delta[0]<0.001 and delta[1]<0.001:
    			return value, happy, sad

# TODO: Fill this function in
# This function implements Policy Improvement. You will 
# use the output of PolicyEvaluation(), and construct a DeterministicPolicy()
# which is an improvement on the old policy and return the new policy
def PolicyImprovement(mdp, value_func, gamma):
	
	states = [mdp.STATE_HAPPY, mdp.STATE_SAD]
	actions = [mdp.ACTION_DRINK, mdp.ACTION_STUDY]
	new_policy = DeterministicPolicy(mdp.get_num_states(), mdp.get_num_actions())
	
	v = [0.0,0.0]
	value = [0.0,0.0]
	
	for i in range(mdp.get_num_states()):
		
		v[i] = value[i]
		best_a = [0,0]
		action_total = [0.0,0.0]

		for ind, a in enumerate(actions):	
			temp = 0
			for j in range(mdp.get_num_states()):
				temp = temp + mdp.get_data(states[i], a, states[j])[0]*mdp.get_data(states[i],a,states[j])[1] + mdp.get_data(states[i], a, states[j])[0]*gamma*value[j]
			
			action_total[ind] = temp
			
			value[i] = max(action_total)
			best_a[i] = actions[np.argmax(action_total)]
			
		new_policy.set_action(states[i], best_a[i])

	policy = new_policy
	
	return policy
    
# TODO: Fill this function in
# This function will take in an old_policy and new_policy
# both of which are of type DeterministicPolicy()
# and return whether PolicyIteration should terminate or not.
def ShouldTerminate(mdp, old_policy, new_policy):
    # return True
    N = mdp.get_num_states()
    
    for state in range(N):
        
        if old_policy.get_action(state) != new_policy.get_action(state):
            return False
    
    return True

# TODO: Fill this function in
# This function makes use of the PolicyEvaluation(), PolicyImprovement(), and ShouldTerminate()
# functions from above and improves on the initial policy. 
# It returns the new improved policy.
def PolicyIteration(mdp, initial_policy, gamma):
	
	termination = True
	value = PolicyEvaluation(mdp, initial_policy, gamma)
	policy = PolicyImprovement(mdp, value, gamma)
	termination = ShouldTerminate(mdp, initial_policy, policy)
	
	while termination == False: 
	
		initial_policy = policy
		termination = ShouldTerminate(mdp, initial_policy, policy)
	
	return policy

# TODO: Fill this function in
# Policy Iteration computes one step of Policy evaluation and one step of policy improvement.
# Value iteration collapses these into one. 
def ValueIteration(mdp, initial_policy, gamma):	
	
	states = [mdp.STATE_HAPPY, mdp.STATE_SAD]
	actions = [mdp.ACTION_DRINK, mdp.ACTION_STUDY]
	new_policy = DeterministicPolicy(mdp.get_num_states(), mdp.get_num_actions())

	v = [0.0,0.0]
	value,_,_ = PolicyEvaluation(mdp, initial_policy, gamma)
	happy = []
	sad = []
	
	while True:
		delta = [0.0,0.0]
		
		for i in range(mdp.get_num_states()):
			
			v[i] = value[i]
			action_total = [0.0,0.0]
			best_a = [0,0]
			
			for ind, a in enumerate(actions):
				temp = 0.0
				for j in range(mdp.get_num_states()):
					temp = temp + mdp.get_data(states[i], a, states[j])[0]*mdp.get_data(states[i],a,states[j])[1] 
					+ mdp.get_data(states[i], a, states[j])[0]*gamma*value[j]

				action_total[ind] = temp
			
			value[i] = max(action_total)
			best_a[i] = actions[np.argmax(action_total)]
			
			if i==0:
				happy = happy + [value[i]]
				#print happy
			if i==1:
				sad = sad+[value[i]]
				#print sad
			delta[i] = max(delta[i], abs(v[i]-value[i]))
			
			if delta[0]<0.001 and delta[1]<0.001:
				for i in range(mdp.get_num_states()):
					new_policy.set_action(states[i], best_a[i])
				return new_policy, value, happy, sad

# TODO: Fill this function in for Question 1.2
def ComputeProbability(initial_state, end_state, time_steps, policy):
	
	mdp = StudentMDP()	
	states = [StudentMDP.STATE_HAPPY, StudentMDP.STATE_SAD]
	num_states = mdp.get_num_states()
	prob_sum = 0
	int_time_steps = time_steps - 1  #int_time_steps: Intermediate time steps (excludes the last time step)
	
	state_combos = set(permutations([StudentMDP.STATE_HAPPY]*num_states + [StudentMDP.STATE_SAD]*num_states, int_time_steps))
	state_combos = list(state_combos)   # generating a t-elements list of combination of states 
	
	for i in range(len(state_combos)):
		
		prob_prod = 1
		temp_initial_state = initial_state
		
		for int_state in state_combos[i]:
			
			prob_prod = prob_prod*mdp.get_data(temp_initial_state, policy.get_action(temp_initial_state), int_state)[0]
			temp_initial_state = int_state

		prob_prod = prob_prod*mdp.get_data(temp_initial_state, policy.get_action(temp_initial_state), end_state)[0]
		prob_sum = prob_sum+prob_prod

	probability = prob_sum

	return probability

if __name__ == '__main__':
    mdp = StudentMDP()
    
    # 1.1
    
    states = [mdp.STATE_HAPPY, mdp.STATE_SAD]
    drink_policy = DeterministicPolicy(mdp.get_num_states(), mdp.get_num_actions())
    study_policy = DeterministicPolicy(mdp.get_num_states(), mdp.get_num_actions())
    
    for state in states:
		
		drink_policy.set_action(state, StudentMDP.ACTION_DRINK)
		study_policy.set_action(state, StudentMDP.ACTION_STUDY)

    # 1
    prob1_drink = ComputeProbability(StudentMDP.STATE_HAPPY, StudentMDP.STATE_HAPPY, 1, drink_policy)
    prob1_study = ComputeProbability(StudentMDP.STATE_HAPPY, StudentMDP.STATE_HAPPY, 1, study_policy)
      
    # 2  
    prob2_drink = ComputeProbability(StudentMDP.STATE_SAD, StudentMDP.STATE_HAPPY, 1, drink_policy)
    prob2_study = ComputeProbability(StudentMDP.STATE_SAD, StudentMDP.STATE_HAPPY, 1, study_policy)
	
    # 3
    prob3_drink = ComputeProbability(StudentMDP.STATE_HAPPY, StudentMDP.STATE_HAPPY, 2, drink_policy)
    prob3_study = ComputeProbability(StudentMDP.STATE_HAPPY, StudentMDP.STATE_HAPPY, 2, study_policy)
    
    # 4
    prob4_drink = ComputeProbability(StudentMDP.STATE_SAD, StudentMDP.STATE_HAPPY, 2, drink_policy)
    prob4_study = ComputeProbability(StudentMDP.STATE_SAD, StudentMDP.STATE_HAPPY, 2, study_policy)
    
    # 5
    prob5_drink = ComputeProbability(StudentMDP.STATE_HAPPY, StudentMDP.STATE_HAPPY, 3, drink_policy)
    prob5_study = ComputeProbability(StudentMDP.STATE_HAPPY, StudentMDP.STATE_HAPPY, 3, study_policy)
    
    # 6
    prob6_drink = ComputeProbability(StudentMDP.STATE_SAD, StudentMDP.STATE_HAPPY, 3, drink_policy)
    prob6_study = ComputeProbability(StudentMDP.STATE_SAD, StudentMDP.STATE_HAPPY, 3, study_policy)
    
    prob_drink = pd.Series([prob1_drink, prob2_drink, prob3_drink, prob4_drink, prob5_drink, prob6_drink])
    prob_study = pd.Series([prob1_study, prob2_study, prob3_study, prob4_study, prob5_study, prob6_study])
    indexes = ['S_1 = H| S_0 = H','S_1 = H| S_0 = S','S_2 = H| S_0 = H','S_2 = H| S_0 = S','S_3 = H| S_0 = H','S_3 = H| S_0 = S']
    columns = ['policy: always drink', 'policy: always study']
    probabilities = pd.concat([prob_drink, prob_study], axis = 1)
    probabilities.index = indexes
    probabilities.columns = columns
    
	# 1.2   
    print probabilities
    
    print "\n"

    # 3.1
    optimal_policy_pi = PolicyIteration(mdp, drink_policy, 0.3)     # Modified the function argument section to take in gamma
    
    converged_value, happy, sad = PolicyEvaluation(mdp, optimal_policy_pi, 0.3)
    print "\n"
    print "Value of the policy from Policy Iteration after convergence", converged_value
    plt.xlabel('Iteration Step Number')
    plt.ylabel('Value')
    plt.title('Optimal Policy from Policy Iteration')
    plt.plot(happy, label = 'Value of State Happy')
    plt.plot(sad, label = 'Value of State Sad')
    plt.legend()
    plt.show()

    # 3.2
    print "Policy Table after Policy Iteration"
    policy_table = pd.DataFrame([optimal_policy_pi.get_action(mdp.STATE_HAPPY), optimal_policy_pi.get_action(mdp.STATE_SAD)], columns = ['Action'], index = ['Happy','Sad'])
    policy_table.index.name = 'State'
    print policy_table
    
    print "\n"

    # 4.1
    optimal_policy_vi, converged_value, happy, sad = ValueIteration(mdp, study_policy, 0.3)
    print "\n" 
    print "Value of the policy from Value Iteration after convergence", converged_value
    plt.xlabel('Iteration Step Number')
    plt.ylabel('Value')
    plt.title('Optimal Policy from Value Iteration')
    plt.plot(happy, label = 'Value of State Happy')
    plt.plot(sad, label = 'Value of State Sad')
    plt.legend()
    plt.show()

    # 4.2
    print "Policy Table after Value Iteration"
    policy_table = pd.DataFrame([optimal_policy_vi.get_action(mdp.STATE_HAPPY), optimal_policy_vi.get_action(mdp.STATE_SAD)], columns = ['Action'], index = ['Happy','Sad'])
    policy_table.index.name = 'State'
    print policy_table