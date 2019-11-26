from DeterministicPolicy import DeterministicPolicy
from PolicyIteration import PolicyEvaluation
import numpy as np

# Value Iteration

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

