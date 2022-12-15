import enum
from json.tool import main
import numpy as np
from scipy.stats import beta, uniform, randint, dirichlet, binom, bernoulli, lognorm, poisson, norm
import math

import matplotlib.pyplot as plt


def upper_bound(data_array, alpha = 0.05, loc = -50, scale = 150):

    return min(100, np.mean(data_array)+scale*math.sqrt(math.log(1/alpha)/2/len(data_array)))

def lower_bound(data_array, alpha = 0.05, loc = -50, scale = 150):

    return np.mean(data_array)-scale*math.sqrt(math.log(1/alpha)/2/len(data_array))

def CI_lower_bound(data_array, alpha = 0.05, loc = -50, scale = 150):

    n = len(data_array)
    dsum = np.sum(data_array)
    sigma = np.std(data_array)

    return dsum/n/(n+1)-4*sigma/(n+1)+dsum/(n+1)-scale*math.sqrt(math.log(1/alpha)/2/(n+1))

def mean_value(data_array, alpha = 0.05, loc = -50, scale = 150):
    
    return min(100, np.mean(data_array))


if __name__ == "__main__":

    data = np.load('Simulation_testing/Simulation_1/data_carla.npy')
    true_action_values = np.load('Simulation_testing/Simulation_1/action_value_carla.npy')
    true_action_value = true_action_values[0]
    
    state_num = 1
    data_size = 50000
    action_num = 30

    data_state_act = np.array([[[]]*action_num]*state_num).tolist()
    
    rule_act = 0
    rate = 0.1
    n_thres = 10

    step_TSRL_value = np.array([[]]*state_num).tolist()
    overall_value = []
    state_data_len = np.array([0]*state_num).tolist()
    TSRL_value = []
    temp = [-50]*action_num
    temp[rule_act] = 100
    TSRL_value = np.array([temp]*state_num).tolist()
    
    last_n = [-1]*action_num
    true_step_TSRL_value = np.array([[]]*state_num).tolist()
    step_TSRL_act = np.array([[]]*state_num).tolist()
    activation_step = np.array([-1]*state_num)
    activation_value = np.array([-1]*state_num)
    
    step_RL_value = []
    RL_value = [uniform.rvs(loc=-50,scale=150,size=action_num) for i in range(state_num)]
    
    true_step_RL_value = []
    step_RL_act = []

    data_values = []

    k = 0
    fig_num = 0


    for idx_ori, state, act_ori, act_value in data[0:20000]:

        data_values.append(act_value)
        
        idx = int(idx_ori)
        act = int(act_ori)
        
        data_state_act[idx][act].append(act_value)

        state_data_len[idx] = state_data_len[idx]+1
        
        k = k+1
          
        if len(data_state_act[idx][act]) > n_thres:
            if act == rule_act:
                TSRL_value[idx][act] = upper_bound(np.array(data_state_act[idx][act]))
            else:
                TSRL_value[idx][act] = min(lower_bound(np.array(data_state_act[idx][act])), CI_lower_bound(np.array(data_state_act[idx][act])))


        step_TSRL_value[idx].append(max(np.array(TSRL_value[idx])))
        TSRL_act = np.argmax(np.array(TSRL_value[idx]))
        step_TSRL_act[idx].append(TSRL_act)
        true_step_TSRL_value[idx].append(true_action_values[idx][TSRL_act])
        
        if activation_step[idx] == -1 and np.argmax(np.array(TSRL_value[idx]))!=rule_act:
            activation_step[idx] = len(step_TSRL_value[idx])

        if k%2000 == 0:
            print(k, TSRL_act, step_TSRL_value[0][-1], true_step_TSRL_value[0][-1])

    plt.figure()

    id = 0 
    print(activation_step[0])
    plt.plot(step_TSRL_value[id],color='black')
    plt.xlim((0,20000))

    plt.show()
    