import numpy as np
from itertools import permutations
import pandas as pd
import matplotlib.pyplot as plt
from PolicyIteration import *#PolicyIteration, PolicyEvaluation
from ValueIteration import ValueIteration
from ComputeProbability import ComputeProbability
from StudentMDP import StudentMDP
from DeterministicPolicy import DeterministicPolicy

def main():    
    
    mdp = StudentMDP()
    
    # 1.1
    
    states = [mdp.STATE_HAPPY, mdp.STATE_SAD]
    drink_policy = DeterministicPolicy(mdp.get_num_states(), mdp.get_num_actions())
    study_policy = DeterministicPolicy(mdp.get_num_states(), mdp.get_num_actions())
    
    for state in states:
		
        drink_policy.set_action(state, mdp.ACTION_DRINK)
        study_policy.set_action(state, mdp.ACTION_STUDY)

    # 1
    prob1_drink = ComputeProbability(mdp, mdp.STATE_HAPPY, mdp.STATE_HAPPY, 1, drink_policy)
    prob1_study = ComputeProbability(mdp, mdp.STATE_HAPPY, mdp.STATE_HAPPY, 1, study_policy)
      
    # 2  
    prob2_drink = ComputeProbability(mdp, mdp.STATE_SAD, mdp.STATE_HAPPY, 1, drink_policy)
    prob2_study = ComputeProbability(mdp, mdp.STATE_SAD, mdp.STATE_HAPPY, 1, study_policy)
	
    # 3
    prob3_drink = ComputeProbability(mdp, mdp.STATE_HAPPY, mdp.STATE_HAPPY, 2, drink_policy)
    prob3_study = ComputeProbability(mdp, mdp.STATE_HAPPY, mdp.STATE_HAPPY, 2, study_policy)
    
    # 4
    prob4_drink = ComputeProbability(mdp, mdp.STATE_SAD, mdp.STATE_HAPPY, 2, drink_policy)
    prob4_study = ComputeProbability(mdp, mdp.STATE_SAD, mdp.STATE_HAPPY, 2, study_policy)
    
    # 5
    prob5_drink = ComputeProbability(mdp, mdp.STATE_HAPPY, mdp.STATE_HAPPY, 3, drink_policy)
    prob5_study = ComputeProbability(mdp, mdp.STATE_HAPPY, mdp.STATE_HAPPY, 3, study_policy)
    
    # 6
    prob6_drink = ComputeProbability(mdp, mdp.STATE_SAD, mdp.STATE_HAPPY, 3, drink_policy)
    prob6_study = ComputeProbability(mdp, mdp.STATE_SAD, mdp.STATE_HAPPY, 3, study_policy)
    
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

if __name__ == '__main__':
    main()