import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import batch_norm


def dqn_model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse): # DQN: Should be the same with each bootstrap model
        out_temp = img_in
        out_temp = tf.layers.dense(out_temp, 128, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.25)) #
        out_temp = tf.layers.dense(out_temp, 128, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.5)) #
        out_temp = tf.layers.dense(out_temp, 128, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.5)) #
        out_temp = tf.layers.dense(out_temp, 128, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.25)) #
        out_temp = tf.layers.dense(out_temp, num_actions, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.25)) #
        return out_temp


def dueling_model(img_in, num_actions, scope, reuse=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)

        return state_score + action_scores

def bootstrap_model(img_in, num_actions, scope, reuse=False, is_training=True):
    """ As described in https://arxiv.org/abs/1602.04621"""
    with tf.variable_scope(scope, reuse=reuse):#, tf.device("/gpu:0"):
        out = img_in
        # with tf.variable_scope("convnet"):
        #     out = tf.layers.dense(out, 128, use_bias=True, activation=None)
        #     out = tf.nn.sigmoid(out)
            
        out_list =[]
        with tf.variable_scope("heads"):
            for _ in range(10):
                scope_net = "action_value_head_" + str(_)
                with tf.variable_scope(scope_net):
                    out_temp = out
                    # out_temp = tf.layers.dense(out_temp, 128, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.25)) #
                    out_temp = tf.layers.dense(out_temp, 256, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.5)) #
                    out_temp = tf.layers.dense(out_temp, 256, activation=tf.nn.relu) #
                    # out_temp = tf.layers.dense(out_temp, 64, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.5)) #

                    out_temp = tf.layers.dense(out_temp, num_actions, activation=None) #
                out_list.append(out_temp)
            
        return out_list


 # out_temp = out
                    # out_temp = tf.layers.dense(out_temp, 128, use_bias=True, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.5))
                    # out_temp = tf.nn.tanh(out_temp)
                    # out_temp = tf.layers.dense(out_temp, 256, use_bias=True, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.5))
                    # out_temp = tf.nn.tanh(out_temp)
                    # out_temp = tf.layers.dense(out_temp, 128, use_bias=True, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.5))
                    # out_temp = tf.nn.tanh(out_temp)