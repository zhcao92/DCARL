import numpy as np
from scipy.stats import beta, uniform, randint, dirichlet, binom, bernoulli, lognorm, poisson, norm
import random

def add_an_act_data(act, action_value):
    
    true_value = action_value[act]
    
    return float(norm.rvs(loc = true_value, scale = 50, size = 1))


def random_state_norm(state_num, size):
    
    states_idxes = norm.rvs(loc=3,scale=1,size=size)
    states_idxes = np.floor(states_idxes/6*state_num).astype(int)
    
    return states_idxes

def random_state_manual(state_num,size):
    
    states_idxes = []
    for i in range(size):
        if random.random() > 0.1:
            states_idxes.append(random.randint(1,state_num-1))
        else:
            states_idxes.append(0)
            
    return states_idxes

def Data_Generation():
    min_value = -50
    max_value = 100

    state_num = 20
    data_size = 50000
    action_num = 11
    action_values = []
    
    states = uniform.rvs(loc=0,scale=1,size=state_num)
    state_times = [0]*state_num
    
    for i in range(state_num):
        action_values.append(uniform.rvs(loc=min_value,scale=max_value-min_value,size=action_num))
    
    states_idxes = random_state_norm(state_num, data_size)
    
    data = []
    
    for idx in states_idxes:
        if idx < 0 or idx >= state_num:
            continue
        state_times[idx] = state_times[idx] + 1
        
        act = random.randint(0,action_num-1)
        data.append([int(idx), states[idx], int(act), add_an_act_data(act, action_values[idx])])
    
    arr = np.hstack((np.expand_dims(np.array(range(state_num)),axis=1), 
                     np.expand_dims(np.array(state_times),axis=1)))
    
    
    sorted_times = arr[np.argsort(-np.array(state_times))]
    
    label = [str(i) for i in sorted_times[:,0]]
    
    np.save('Simulation_testing/Simulation_Data_Collection/data.npy',np.array(data))
    np.save('Simulation_testing/Simulation_Data_Collection/action_value.npy',np.array(action_values))
    np.save('Simulation_testing/Simulation_Data_Collection/states.npy',np.array(states))


if __name__ == "__main__":

    Data_Generation()

    