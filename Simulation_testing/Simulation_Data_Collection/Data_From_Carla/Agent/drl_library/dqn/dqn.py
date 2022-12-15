
import math, random
from socket import TIPC_SRC_DROPPABLE

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from Planning_library.trustset import Trustset

from Agent.drl_library.dqn.replay_buffer import NaivePrioritizedBuffer, Replay_Buffer
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Q_network(nn.Module):
    def __init__(self, num_inputs, num_actions, global_graph_width=3):
        super(Q_network, self).__init__()
        
        self.q_lin = nn.Linear(num_inputs, global_graph_width)
        self.k_lin = nn.Linear(num_inputs, global_graph_width)
        self.v_lin = nn.Linear(num_inputs, global_graph_width)

        self._norm_fact = 1 / math.sqrt(num_inputs)
        
        self.layers = nn.Sequential(
            nn.Linear(global_graph_width, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.num_actions = num_actions
        
    def forward(self, x):
        n = int(len(x[0])/5)
        x = torch.reshape(x, [n,5]).unsqueeze(0)

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2)) * self._norm_fact
        scores = nn.functional.softmax(scores, dim=-1)
        
        atten_result =  torch.bmm(scores,value)[0][0]
        return self.layers(atten_result)
    
    def act_hybrid(self, state, TS, action_num = 8):
        """
        Main hybrid
        """

        TS_indexes = []

        for act in range(action_num):
            TS_indexes.append(TS.TS_ConfidenceValue(state, act))
        
        return np.argmax(np.array(TS_indexes))

    def ego_attention(self, x):
        # x contains n vehicle
        # query, key, value contain n number
        
        n = int(len(x[0])/5)
        x = torch.reshape(x, [n,5]).unsqueeze(0)

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        ego_scores = torch.bmm(query[0][0].unsqueeze(0).unsqueeze(0), key.transpose(1, 2)) * self._norm_fact
        ego_scores = nn.functional.softmax(ego_scores, dim=-1)
        ego_atten_result = torch.mul(ego_scores.transpose(2,1), value)
        
        return ego_atten_result
        
    def encoded_state(self, x):
        n = int(len(x[0])/5)
        x = torch.reshape(x, [n,5]).unsqueeze(0)

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2)) * self._norm_fact
        scores = nn.functional.softmax(scores, dim=-1)
        
        atten_result =  torch.bmm(scores,value)[0][0]
                
        return atten_result
    
    def act_ts(self, state, TS, action_num = 3):
        
        state_np = state
        state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state).unsqueeze(0)
        q_value_np = q_value.detach().numpy()[0]
        
        for act in range(action_num):
            if not TS.in_TS(state_np,np.array(act)):
                q_value_np[act] = -1000
                
        print(q_value_np)
        action = np.argmax(q_value_np)
        return action
        
    def act_ts_explore(self, state, encoded_state, TS, action_num = 3, c = 5):
        
        state_np = state
        state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state).unsqueeze(0)
        q_value_np = q_value.detach().numpy()[0]
        
        N_a = TS.get_state_num(encoded_state)
        act_value = []
        for act in range(action_num):
            if N_a[act] == 0:
                N_a[act] = 1
            act_value.append(q_value_np[act]+c*math.sqrt(math.log(sum(N_a))/N_a[act]))
        
        action = np.argmax(np.array(act_value))
        print(action, np.around(act_value,1), np.around(q_value_np,1), N_a)
        
        return action
        
    def act(self, state, epsilon, N_a, action_num = 3, c = 5):
        
        state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state).unsqueeze(0)
        # print("q_value", q_value.unsqueeze(0))
        # q_value_np = q_value.detach().numpy()[0]
        # print(q_value_np, q_value_np[0])
        # for act in range(action_num):
        #     act_value.append(q_value_np[act]+c*math.sqrt(math.log(sum(N_a))/N_a[act]))
        
        # action = np.argmax(np.array(act_value))
        # print("action:", action, act_value, q_value_np)
        
        # action  = q_value.max(1)[1].data[0]
        if random.random() > epsilon:
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        
        return action
    
class DQN():
    def __init__(self, env, batch_size):
        self.env = env
        self.current_model = Q_network(5, env.action_space.n)
        # self.current_model = Q_network(env.observation_space.shape[0], env.action_space.n)

        self.target_model  = Q_network(5, env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = NaivePrioritizedBuffer(1000000)
        self.TS = Trustset()
        # self.replay_buffer = Replay_Buffer(obs_shape=env.observation_space.shape,
        #     action_shape=env.action_space.shape, # discrete, 1 dimension!
        #     capacity= 1000000,
        #     batch_size= self.batch_size,
        #     device=self.device)
        
    def compute_td_loss(self, batch_size, beta, gamma):
        for i in range(batch_size):
            
            state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(1, beta) 
            
            encoded_state = self.get_encoded_state(state)
            encoded_next_state = self.get_encoded_state(next_state)
            
            self.TS.add_data(encoded_state, np.array(action), np.array(reward))
            
            no_data_punishment = -10
            if self.TS.in_TS(encoded_next_state):
                no_data_punishment = 0
            
            state      = Variable(torch.FloatTensor(np.float32(state)))
            next_state = Variable(torch.FloatTensor(np.float32(next_state)))
            action     = Variable(torch.LongTensor(action))
            reward     = Variable(torch.FloatTensor(reward))
            done       = Variable(torch.FloatTensor(done))
            weights    = Variable(torch.FloatTensor(weights))
            q_values      = self.current_model(state)
            
            next_q_values = self.target_model(next_state)
            q_value          = (q_values.unsqueeze(0)).gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value     = next_q_values.unsqueeze(0).max(1)[0]
            
            expected_q_value = reward + gamma * next_q_value * (1 - done) + no_data_punishment
            # print(next_q_value, expected_q_value)
            loss  = (q_value - expected_q_value.detach()).pow(2) * weights
            prios = loss + 1e-5
            loss  = loss.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.optimizer.step()
        
        return loss
    
    def get_encoded_state(self, state):
        
        state = Variable(torch.FloatTensor(np.float32(state)))
        
        return self.target_model.encoded_state(state).detach().numpy()
    
    def get_trained_value_sa(self, state, action):
        
        state      = Variable(torch.FloatTensor(np.float32([state])))
        action     = Variable(torch.LongTensor([action]))
        
        q_value_s = self.current_model(state).unsqueeze(0)        
        q_value_sa = q_value_s.gather(1, action.unsqueeze(1)).squeeze(1)
        return q_value_sa
    
    def get_trained_value_s(self, state):
        state      = Variable(torch.FloatTensor(np.float32([state])))
        
        q_value = self.current_model(state)
        return q_value
        
    def get_expected_value(self, state, action, reward, next_state, done, gamma):
        
        
        state      = Variable(torch.FloatTensor(np.float32([state])))
        next_state = Variable(torch.FloatTensor(np.float32([next_state])))
        action     = Variable(torch.LongTensor([action]))
        reward     = Variable(torch.FloatTensor([reward]))
        done       = Variable(torch.FloatTensor([done]))
                
        next_q_value = (self.target_model(next_state).unsqueeze(0)).max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        return expected_q_value
    
    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 0.9
        epsilon_final = 0.1
        epsilon_decay = 1000000
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    
    def beta_by_frame(self, frame_idx):
        beta_start = 0.4
        beta_frames = 1000  
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    
    def train(self, load_step, num_frames, gamma):
        losses = []
        all_rewards = []
        episode_reward = 0
        
        self.load(load_step)
        
        # Create Agent
        trajectory_planner = JunctionTrajectoryPlanner()
        controller = Controller()
        dynamic_map = DynamicMap()
        r_thres = 1
        pass_thres = 10

        obs, obs_ori = self.env.reset()
        dqn_action = None
        sum_reward = 0
        passed_data = []
        frame_idx = load_step
        key_data = None
        
        # while frame_idx < load_step+num_frames + 1:
        #     frame_idx = frame_idx + 1
        #     obs = np.array(obs)
        #     encoded_obs = self.get_encoded_state(np.array([obs]))
        #     # epsilon = self.epsilon_by_frame(frame_idx)
        #     if dqn_action is None:
        #         # dqn_action = self.current_model.act(obs, epsilon, N_a)
        #         # dqn_action = self.current_model.act_ts(obs, self.TS)
        #         # dqn_action = self.current_model.act_ts_explore(obs, self.TS)
        #         dqn_action = self.current_model.act_ts_explore(obs, encoded_obs, self.TS)

        #     obs_tensor = Variable(torch.FloatTensor(np.float32(obs))).unsqueeze(0)
        #     ego_attention = self.current_model.ego_attention(obs_tensor).detach().numpy()
                        
        #     obs_ori = np.array(obs_ori)
        #     dynamic_map.update_map_from_obs(obs_ori, self.env)
        #     rule_trajectory, action = trajectory_planner.trajectory_update(dynamic_map)
        #     rule_trajectory = trajectory_planner.trajectory_update_CP(dqn_action, rule_trajectory)
        #     control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
        #     action = [control_action.acc, control_action.steering]
            
        #     new_obs, reward, done, new_obs_ori = self.env.step(action, ego_attention = ego_attention)
        #     passed_data.append([obs, reward])
        #     sum_reward = sum_reward+reward
            
        #     obs = new_obs
        #     obs_ori = new_obs_ori
            
        #     episode_reward += reward
        
        #     if sum_reward > r_thres or len(passed_data)>pass_thres or done:
        #         for data in passed_data:
        #             self.replay_buffer.push(data[0], dqn_action, sum_reward, new_obs, done)
        #             sum_reward = sum_reward - data[1]
        #         # print("------")
        #         # replay_buffer_len = len(self.replay_buffer)
        #         # passed_data_len = len(passed_data)
        #         # print(self.replay_buffer.get(replay_buffer_len-passed_data_len))
                
        #         # key_data = self.replay_buffer.get(replay_buffer_len-passed_data_len)
        #         # print("before training:", key_data[0], key_data[1])
        #         # print(self.get_trained_value_sa(key_data[0], key_data[1]))
        #         # print(self.get_expected_value(key_data[0],key_data[1],key_data[2],key_data[3],key_data[4],gamma))
                
        #         dqn_action = None
        #         sum_reward = 0
        #         passed_data = []
                
        #         if done:
        #             obs, obs_ori = self.env.reset()
        #             trajectory_planner.clear_buff(clean_csp=True)
        #             all_rewards.append(episode_reward)
        #             episode_reward = 0
                    
        #     if (frame_idx) > self.batch_size:
        #         beta = self.beta_by_frame(frame_idx)
        #         loss = self.compute_td_loss(self.batch_size, beta, gamma)
            
        #     if (frame_idx) % 10000 == 0:
        #         self.update_target(self.current_model, self.target_model)
        #         self.save(frame_idx)
            
        #     # if key_data is not None:
        #     #     print("after training:")
        #     #     print(self.get_trained_value_sa(key_data[0], key_data[1]))
        #     #     key_data = None
                
            
        for frame_idx in range(load_step, load_step+num_frames + 1):
            
            dqn_action = None
            sum_reward = 0
            passed_data = []
            
            while True:
                obs = np.array(obs)
                epsilon = self.epsilon_by_frame(frame_idx)
                if dqn_action is None:
                    dqn_action = self.current_model.act(obs, epsilon)
                obs_tensor = Variable(torch.FloatTensor(np.float32(obs))).unsqueeze(0)
                ego_attention = self.current_model.ego_attention(obs_tensor).detach().numpy()
                
                obs_ori = np.array(obs_ori)
                dynamic_map.update_map_from_obs(obs_ori, self.env)
                rule_trajectory, action = trajectory_planner.trajectory_update(dynamic_map)
                rule_trajectory = trajectory_planner.trajectory_update_CP(dqn_action, rule_trajectory)
                control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
                action = [control_action.acc, control_action.steering]
                
                new_obs, reward, done, new_obs_ori = self.env.step(action, ego_attention = ego_attention)
                passed_data.append([obs, reward])
                sum_reward = sum_reward+reward
                
                obs = new_obs
                obs_ori = new_obs_ori
            
                if sum_reward > r_thres or len(passed_data)>pass_thres or done:
                    for data in passed_data:
                        self.replay_buffer.push(data[0], dqn_action, sum_reward, new_obs, done)
                        sum_reward = sum_reward - data[1]
                    break
            
            new_obs, reward, done, new_obs_ori = self.env.step(action, ego_attention = ego_attention)
            print("[DQN]: ----> RL Action",dqn_action)

            # self.replay_buffer.add(obs, np.array([dqn_action]), np.array([reward]), new_obs, np.array([done]))
            self.replay_buffer.push(obs, dqn_action, reward, new_obs, done)
            
            obs = new_obs
            
            episode_reward += reward
            
            if done:
                obs = self.env.reset()
                trajectory_planner.clear_buff(clean_csp=True)

                all_rewards.append(episode_reward)
                episode_reward = 0
                
            if (frame_idx) > self.batch_size:
                beta = self.beta_by_frame(frame_idx)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                
                losses.append(loss.data[0])
                
            if frame_idx % 200 == 0:
                plot(frame_idx, all_rewards, losses)
                
            if (frame_idx) % 10000 == 0:
                self.update_target(self.current_model, self.target_model)
                self.save(frame_idx)

    def save(self, step):
        torch.save(
            self.current_model.state_dict(),
            'saved_model/current_model_%s.pt' % (step)
        )
        torch.save(
            self.target_model.state_dict(),
            'saved_model/target_model_%s.pt' % (step)
        )
        torch.save(
            self.replay_buffer,
            'saved_model/replay_buffer_%s.pt' % (step)
        )
        
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