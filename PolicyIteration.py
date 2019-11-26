import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from StudentMDP import StudentMDP
from DeterministicPolicy import DeterministicPolicy

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

# Policy Iteration computes one step of Policy evaluation and one step of policy improvement.
# Value iteration collapses these into one. 
