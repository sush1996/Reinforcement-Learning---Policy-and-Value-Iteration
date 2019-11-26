from itertools import permutations

# Compute probabilties of starting from some state at time 0 and reaching another state at time t  

def ComputeProbability(mdp, initial_state, end_state, time_steps, policy):
	
	#mdp = StudentMDP()	
	states = [mdp.STATE_HAPPY, mdp.STATE_SAD]
	num_states = mdp.get_num_states()
	prob_sum = 0
	int_time_steps = time_steps - 1  #int_time_steps: Intermediate time steps (excludes the last time step)
	
	state_combos = set(permutations([mdp.STATE_HAPPY]*num_states + [mdp.STATE_SAD]*num_states, int_time_steps))
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