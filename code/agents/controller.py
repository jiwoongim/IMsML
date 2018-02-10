import os, time, random
import numpy as np
import tensorflow as tf

from collections import deque
from utils.nn_utils import linear_annealing
rng = np.random.RandomState(1234)

##TODO 
class Controller(object):
    ''' Q-Agent '''

    def __init__(self, config, model, agent, optim, session, \
                        scope_name='agent', summary_writer=None):

        self.agent          = agent
        self.model          = model
        self.optim          = optim
        self.session        = session
        self.scope_name     = scope_name+'_controller'
        self.epsilon        = config['epsilon']           #epsilon-greedy
        self.num_actions    = config['num_actions']
        self.D              = config['D']
        self.batch_sz       = config['batch_sz']


        self.exploration_period = config['exploration_period']
        self.store_freq         = config['store_freq']
        self.train_freq         = config['train_freq']
        self.tq_update_freq     = config['tq_update_freq']
        self.max_experience     = config['max_experience']
        #self.target_network_update_rate = \
        #        tf.constant(config['Qt_network_lr'])


        # DQN state
        self.num_updates    = 0 # Number of updates 
        self.num_act_exec   = 0
        self.experience     = deque()
        self.summary_writer = summary_writer

        self.num_store = 0
        self.num_steps = 0

        #Initialize target_network
        #self.target_network_update()


    def take_action(self, states, epsilon=0.05):
        ''' Agent applies action and moves into new state '''

        self.num_act_exec += 1 
        if epsilon is None: epsilon = self.epsilon
        eps = linear_annealing(self.num_act_exec,
                                    self.exploration_period,
                                    1.0, epsilon)

        if rng.random_sample() < eps:
            return rng.randint(0, self.num_actions - 1)
        else:
            return self.session.run(self.agent.get_action(self.agent.states)[1], \
                    {self.agent.states: states[np.newaxis,:]})[0]


    def store(self, state, action, reward, new_state):
        """Store experience, where starting with state and
        execution action, we arrived at the new state and got thetarget_network_update
        reward reward

        If newstate is None, the state/action pair is assumed to be terminal

        Code from https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/discrete_deepq/
        """

        if self.num_store % self.store_freq == 0:
            self.experience.append((state, action, reward, new_state))
            if len(self.experience) > self.max_experience:
                self.experience.popleft()
        self.num_store += 1


    def step(self):

        self.num_steps += 1

        if (self.num_steps % self.train_freq != 0) \
                or (len(self.experience) < self.batch_sz):
            
            return 

        # Sample a batch data
        samples, states, new_states, action_mask, new_states_mask, rewards = \
                                    init_batch(self.experience, \
                                    len(self.experience), \
                                    self.batch_sz, self.D, self.num_actions)

        calculate_summaries = self.num_updates % 100 == 0 and \
                self.summary_writer is not None

        ## List of functions need to be evaluated
        fevals = [  self.agent.cost(self.agent.pred_q, self.agent.future_rewards),\
                    self.optim,\
                    self.agent.summarize] 

        # Evaluate functions
        cost, _, summary_str = self.session.run(fevals, {\
                self.agent.states : states, \
                self.agent.next_states : new_states,\
                self.agent.next_states_mask : new_states_mask,\
                self.agent.action_mask: action_mask,\
                self.agent.rewards : rewards\
                })

        # Update target functions 
        # TODO : Need to evaluate every few steps, not every step.
        if self.num_updates % self.tq_update_freq == self.tq_update_freq-1:
            self.agent.update_tq()

        if calculate_summaries:
            self.summary_writer.add_summary(summary_str, self.num_updates)
            print("Num Updates %d, Train Cost %f" % (self.num_updates, cost))
        self.num_updates += 1

        return cost

def init_batch(experiences, N, n, D, C):

    ## Experience replay.
    samples_ind = rng.randint(N, size=n)
    samples     = [experiences[index] for index in samples_ind]


    ## Make batch 
    states         = np.empty((n, D), dtype='float32')
    new_states     = np.empty((n, D), dtype='float32')
    action_mask    = np.zeros((n, C), dtype='float32')

    new_states_mask = np.empty((n,), dtype='float32')
    rewards         = np.empty((n,), dtype='float32')

    ## Filling in the state
    ## TODO : Is this the most efficient way to set the sample states?
    ## Here, we have to iterate over the samples for every update.
    for i, (state, action, reward, newstate) in enumerate(samples):

        states[i] = state
        action_mask[i] = 0
        action_mask[i][action] = 1
        rewards[i] = reward
        if newstate is not None:
            new_states[i] = newstate
            new_states_mask[i] = 1
        else:
            new_states[i] = 0
            new_states_mask[i] = 0

    return samples, states, new_states, action_mask, new_states_mask, rewards


