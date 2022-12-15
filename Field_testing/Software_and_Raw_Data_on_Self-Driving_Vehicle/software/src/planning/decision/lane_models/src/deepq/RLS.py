import os
import os.path as osp
import random
from rtree import index as rindex

class RLS(object):

    def __init__(self,
                 visited_times_thres = 30,
                 is_training = True,
                 debug = True,
                 save_new_data = True,
                 create_new_train_file = True,
                 create_new_record_file = True,
                 save_new_driving_data = True):

        self.visited_times_thres = visited_times_thres
        self.is_training = is_training
        self.trajectory_buffer = deque(maxlen=20)
        self.debug = debug
        self.save_new_data = save_new_data
        self.create_new_train_file = create_new_train_file
        self.create_new_record_file = create_new_record_file
        self.save_new_driving_data = save_new_driving_data
        self._setup_data_saving()
        
    def _setup_data_saving(self):

        if self.create_new_train_file:
            if osp.exists("state_index.dat"):
                os.remove("state_index.dat")
                os.remove("state_index.idx")
            if osp.exists("visited_state.txt"):
                os.remove("visited_state.txt")
            if osp.exists("visited_value.txt"):
                os.remove("visited_value.txt")

            self.visited_state_value = []
            self.visited_state_counter = 0
        else:
            self.visited_state_value = np.loadtxt("visited_value.txt")
            self.visited_state_value = visited_state_value.tolist()
            self.visited_state_counter = len(visited_state_value)

        self.visited_state_outfile = open("visited_state.txt", "a")
        self.visited_state_format = " ".join(("%f",)*14)+"\n"

        self.visited_value_outfile = open("visited_value.txt", "a")
        self.visited_value_format = " ".join(("%f",)*2)+"\n"

        visited_state_tree_prop = rindex.Property()
        visited_state_tree_prop.dimension = 13
        self.visited_state_dist = np.array([[2, 5, 10, 1, 6, 10, 1, 6, 10, 6, 10, 6, 0.1]])
        self.visited_state_tree = rindex.Index('state_index',properties=visited_state_tree_prop)

        if create_new_record_file:
            if osp.exists("driving_record.txt"):
                os.remove("driving_record.txt")
        self.driving_record_outfile = open("driving_record.txt","a")
        self.driving_record_format = " ".join(("%f",)*22)+"\n"

    def act(self, obs, RL_action):
        if self.is_training:
            return self.act_train(obs, RL_action)
        else:
            return self.act_test(obs, RL_action)
    
    def act_train(self,obs, RL_action):
        if self.should_use_rule(obs): 
            return 0
        else:
            # epsilon greedy from DQN
            return RL_action
    
    def state_with_action(self,obs,action):
        return obs.append(action)

    def should_use_rule(self,obs):
        """
        Whether the state should use rule action
        """
        # Rule action not explore enough

        rule_state = self.state_with_action(obs,0)
        visited_times_rule = self._calculate_visited_times(rule_state,self.visited_state_tree)
        mean_rule, var_rule, sigma_rule = self._calculate_statistics_index(rule_state,
                                                                    self.visited_state_value,
                                                                    self.visited_state_tree) 

        if debug:
            print("rule visited times:",visited_times_rule)
            print("current state:", obs)
            print("---------------------")

        if visited_times_rule < self.visited_times_thres:
            return True

        # Rule perform good
        # mean_rule in (-1,0)
        explore_motivation = random.uniform(-1,0)
        if explore_motivation < mean_rule:
            return True
        return False

    def act_test(self, obs, RL_action):
        if RL_action == 0:
            return 0
        
        rule_state = self.state_with_action(obs,0)
        visited_times_rule = self._calculate_visited_times(rule_state,self.visited_state_tree)
        mean_rule, var_rule, sigma_rule = self._calculate_statistics_index(rule_state,
                                                                    self.visited_state_value,
                                                                    self.visited_state_tree) 
        for candidate_action in range(1,8):

            RL_state = self.state_with_action(obs,candidate_action)
            visited_times_RL = self._calculate_visited_times(RL_state,self.visited_state_tree)
            mean_RL, var_RL, sigma_RL = self._calculate_statistics_index(RL_state,self.visited_state_value,self.visited_state_tree)

            if visited_times_rule < self.visited_times_thres or visited_times_RL < 10 or mean_rule > -0.1:
                continue
        
            var_diff = var_rule/visited_times_rule + var_RL/visited_times_RL
            sigma_diff = np.sqrt(var_diff)
            mean_diff = mean_RL - mean_rule

            z = mean_diff/sigma_diff
            # print(action,norm.cdf(z))
            if norm.cdf(z)>confidence_thres:
                return candidate_action
        
        return 0

    ############## RLS Confidence ##############

    def _calculate_visited_times(self, obs, visited_state_tree):

    return sum(1 for _ in visited_state_tree.intersection(obs.tolist()))

    def _calculate_statistics_index(self, obs, visited_state_value, visited_state_tree):
        """
        Calculate statistics_idx
        """
        if _calculate_visited_times(obs,visited_state_tree, max_idx) == 0:
            return -1, -1, -1

        value_list = [visited_state_value[idx] for idx in visited_state_tree.intersection(obs.tolist())]
        value_array_av = np.array(value_list)
        value_array = value_array_av[:,1]
        # value_array_rule = value_array[value_array[:,0]==0][:,1]
        # value_array_RL = value_array
        mean = np.mean(value_array)
        var = np.var(value_array)
        sigma = np.sqrt(var)

        return mean,var,sigma

    ############## DATASET ##############

    def add_data(self, obs, action, rew, new_obs, done):
        self.trajectory_buffer.append(obs, action, rew, new_obs, done)

        while len(self.trajectory_buffer) > 10:
            obs_left, action_left, rew_left, new_obs_left, done_left = trajectory_buffer.popleft()
            state_to_record = self.state_with_action(obs_left, action_left)
            action_to_record = action_left
            r_to_record = rew_left
            if self.save_new_data:
                self.visited_state_value.append([action_to_record,r_to_record])
                self.visited_state_tree.insert(self.visited_state_counter,
                    tuple((state_to_record-self.visited_state_dist).tolist()[0]+(state_to_record+self.visited_state_dist).tolist()[0]))
                self.visited_state_outfile.write(self.visited_state_format % tuple(state_to_record[0]))
                self.visited_value_outfile.write(self.visited_value_format % tuple([action_to_record,r_to_record]))
                self.visited_state_counter += 1
        

        if done:
            _, _, rew_right, _, _ = trajectory_buffer[-1]
            while len(trajectory_buffer)>0:
                obs_left, action_left, rew_left, new_obs_left, done_left = trajectory_buffer.popleft()
                action_to_record = action_left
                r_to_record = rew_right*gamma**len(trajectory_buffer)
                state_to_record = self.state_with_action(obs_left, action_left)
                if self.save_new_data:
                    self.visited_state_value.append([action_to_record,r_to_record])
                    self.visited_state_tree.insert(self.visited_state_counter,
                        tuple((state_to_record-self.visited_state_dist).tolist()[0]+(state_to_record+self.visited_state_dist).tolist()[0]))
                    self.visited_state_outfile.write(self.visited_state_format % tuple(state_to_record[0]))
                    self.visited_value_outfile.write(self.visited_value_format % tuple([action_to_record,r_to_record]))
                    self.visited_state_counter += 1

        if self.save_new_driving_data:
            state_rule = self.state_with_action(obs,0)
            visited_times_rule = _calculate_visited_times(state_rule,self.visited_state_rule_tree)
            mean_rule, var_rule, sigma_rule = _calculate_statistics_index(state_rule,self.visited_state_rule_value,self.visited_state_rule_tree)
            if action == 0:
                visited_times_RL = -1
                mean_RL = -1
                var_RL = -1
            else:
                RL_state = self.state_with_action(obs,action)
                visited_times_RL = _calculate_visited_times(RL_state,self.visited_state_tree)
                mean_RL, var_RL, sigma_RL = _calculate_statistics_index(RL_state,self.visited_state_value,visited_state_tree)

            record_data = state_rule
            record_data = np.append(record_data,rew)
            record_data = np.append(record_data,float(done))
            record_data = np.append(record_data,visited_times_rule)
            record_data = np.append(record_data,mean_rule)
            record_data = np.append(record_data,var_rule)
            record_data = np.append(record_data,visited_times_RL)
            record_data = np.append(record_data,mean_RL)
            record_data = np.append(record_data,var_RL)

            self.driving_record_outfile.write(self.driving_record_format % tuple(record_data))


    
