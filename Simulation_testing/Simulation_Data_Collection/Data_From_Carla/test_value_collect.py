from Test_Scenarios.TestScenario_Town03 import CarEnv_Town03_Complex

from Agent.drl_library.dqn.dqn_value_collect import DQNTest

EPISODES=2642

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_Town03_Complex()

    model = DQNTest(env, batch_size=20)
    model.test(load_step=300000, num_frames=300000,  gamma=0.99)