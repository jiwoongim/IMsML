import os, time, random
import numpy as np
import tensorflow as tf

from agent import Agent

rng = np.random.RandomState(1234)

class QAgent(Agent):
    ''' Q-Agent '''


    def __init__(self, q_config, model, scope_name):
        super(QAgent, self).__init__(q_config)
    
        self.q_network      = model
        self.scope_name     = scope_name
        self.epsilon        = q_config['epsilon']           #epsilon-greedy
        self.discount_rate  = q_config['discount_rate']
        self.num_actions    = q_config['num_actions']
        self.D              = q_config['D']

        self.tq_network     = self.q_network.clone(new_scope_name='tq_network')  #Target Q-Network 
        self.J              = self._build_graph()


    def _build_graph(self):
        '''Build model'''
       
        with tf.name_scope("take_action"):
        
            #TODO Currently can only handles 2D states. We may need to consider 
            # 3D state cases. For example, sequences of frames for atari game.
            self.states =  tf.placeholder(tf.float32, (None, self.D), name='state')
            self.action_scores, pred_actions = self.get_action(self.states)

        with tf.name_scope("get_future_reward"):

            self.next_states        = tf.placeholder(tf.float32, \
                                        (None, self.D), name="next_state")
            self.next_states_mask   = tf.placeholder(tf.float32, (None,),\
                                        name="next_state_mask")


        with tf.name_scope("predict_q"):

            self.action_mask = tf.placeholder(tf.float32, \
                                (None, self.num_actions), name="action_mask")
            self.pred_q = self.predict_q(self.action_scores, self.action_mask)

        with tf.name_scope("future_rewards"):

            self.rewards    = tf.placeholder(tf.float32, (None,), name="rewards")
            self.future_rewards  = self.get_future_reward(\
                                        self.next_states, 
                                        self.next_states_mask, \
                                        self.action_scores, \
                                        self.rewards)

        self.J = self.cost(self.pred_q, self.future_rewards)

        ## Histogram 
        with tf.variable_scope('summary'):
            tf.histogram_summary("action_scores", self.action_scores)
            tf.histogram_summary("prediction_error", self.pred_q)
            tf.histogram_summary("target_action_scores", self.future_rewards)
            tf.scalar_summary("Cost", self.J)
            self.summarize = tf.merge_all_summaries() 
            # TODO : Histogram of average Action Chosen


    def get_action(self, states):
        '''Returns a action with the maximum action score and actions scores.'''
        
        #action_scores = tf.identity(self.q_network.fp(states), name="action_scores")
        action_scores = self.q_network.fp(states)
        pred_actions  = tf.argmax(action_scores, dimension=1)

        return action_scores, pred_actions


    def get_future_reward(self, new_state, state_mask, action_scores, rewards):
        '''Returns the target Q value'''

        print self.tq_network
        # Stop_gradient makes the score be a constant so that gradient doesn't flow
        new_action_scores   = tf.stop_gradient(self.tq_network.fp(new_state)) 
        target_values       = tf.reduce_max(new_action_scores, reduction_indices=[1,])\
                                    * state_mask

        return rewards + self.discount_rate * target_values


    def predict_q(self, action_scores, action_mask):
        '''Returns the q-score of the predicted action'''

        return tf.reduce_sum(action_scores * action_mask, reduction_indices=[1,])
        


    def cost(self, pred_q, target_q):
        ''' Objective function '''

        return tf.reduce_mean(tf.square(pred_q - target_q), name='cost')

    
    def update_tq(self):
        '''Update target Q-network'''

        # TODO : Perhaps it is good to make into tf.group ??
        # self.tq_network = self.q_network.clone(new_scope_name='tq_network')  
        for q_param, tq_param in zip(self.q_network.params, self.tq_network.params):

            tq_param.assign(q_param.eval())



