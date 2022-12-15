
import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import numpy as np

from Agent.drl_library.dqn.replay_buffer import NaivePrioritizedBuffer
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap

USE_CUDA = False#torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Q_network(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Q_network, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.num_actions = num_actions
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action
    
    def act_hybrid(self, state, rule_action, TS):
        
        act, trust_RL = TS.hybrid_act(state, rule_action)
        
        return act, trust_RL
    
class DQNTest():
    def __init__(self, env, batch_size):
        self.env = env
        self.current_model = Q_network(env.observation_space.shape[0], env.action_space.n)
        self.target_model  = Q_network(env.observation_space.shape[0], env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = NaivePrioritizedBuffer(1000000)
            
    def test(self, load_step, num_frames, gamma):
        all_rewards = []
        episode_reward = 0
        
        self.load(load_step)
        
        HRL_triggered = False
        trust_RL = False
        
        # Create Agent
        trajectory_planner = JunctionTrajectoryPlanner()
        controller = Controller()
        dynamic_map = DynamicMap()

        obs, obs_ori = self.env.reset()        
        used_action = 0
        
        action_value = []

        while True:
            obs_ori = np.array(obs_ori)
            obs = np.array(obs)
            
            dynamic_map.update_map_from_obs(obs_ori, self.env)

            candidate_trajectories = trajectory_planner.get_all_candidate_trajectories(dynamic_map)
            rule_index = trajectory_planner.get_optimal_trajectory(candidate_trajectories)
            final_trajectory = trajectory_planner.get_trajectory_by_index(rule_index, candidate_trajectories) 
            
            if not HRL_triggered and obs_ori[1] < 90:
                hybrid_action = used_action
                final_trajectory = trajectory_planner.get_trajectory_by_index(hybrid_action, candidate_trajectories)
                HRL_triggered = True
                HRL_trajectory = final_trajectory
                recorded_state = obs_ori
            
            trajectory_planner.update_last_trajectory(final_trajectory)
            
            if HRL_triggered:
                used_final_trajectory = HRL_trajectory
                self.env.draw_traj([[HRL_trajectory]])
            else:
                used_final_trajectory = final_trajectory
                
            converted_trajectory = trajectory_planner.convert_trajectory_to_TAction(used_final_trajectory)
            control_action = controller.get_control(dynamic_map,  converted_trajectory.trajectory, converted_trajectory.desired_speed)
            action = [control_action.acc, control_action.steering]
            
            new_obs, reward, done, new_obs_ori = self.env.step(action, trust_RL)

            obs = new_obs
            obs_ori = new_obs_ori
            episode_reward += reward

            if done:
                AveSpeed = sum(self.env.driving_speed)/len(self.env.driving_speed)
                obs, obs_ori = self.env.reset()
                trajectory_planner.clear_buff(clean_csp=True)

                all_rewards.append(episode_reward)
                
                fw = open("collected_data.txt", 'a')
                # Write num
                fw.write(str(recorded_state)) 
                fw.write(", ")
                fw.write(str(used_action)) 
                fw.write(", ")
                fw.write(str(episode_reward)) 

                fw.write("\n")
                fw.close()                               
                
                action_value.append([used_action, episode_reward, AveSpeed])
                episode_reward = 0
                HRL_triggered = False
                trust_RL = False
                
                used_action = used_action+1
                used_action = used_action%(len(candidate_trajectories)+1)
                
        
    def load(self, load_step):
        
        try:
            self.current_model.load_state_dict(
            torch.load('saved_model/current_model_%s.pt' % (load_step))
            )

            self.target_model.load_state_dict(
            torch.load('saved_model/target_model_%s.pt' % (load_step))
            )
            
            self.replay_buffer = torch.load('saved_model/replay_buffer_%s.pt' % (load_step))        
                
            print("[DQN] : Load learned model successful, step=",load_step)
        
        except:
            
            load_step = 0
            
            print("[DQN] : No learned model, Creat new model")
            
        return load_step