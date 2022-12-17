# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:20:33 2022

@author: mosta
"""

#!/usr/bin/env python
# coding: utf-8

# In[42]:


from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# In[44]:

class Building():
    
    def __init__(self, maximum_ventilation_power,time_step_size, 
                 CO2_setpoint, initial_time_step, action_space):

		# CONSTANT VALUES
        self.__maximum_ventilation_power = maximum_ventilation_power
        self.__time_step_size = time_step_size
        self.initial_time_step = initial_time_step
        self.CO2_setpoint = CO2_setpoint

		# VARYING VALUES
        self.current_ventilation_power = None
        self.current_int_CO2 = None
        self.current_time_interval = None
        self.action_space = action_space
        
        
    def initialize_state(self, initial_int_CO2):
        self.current_int_CO2 = initial_int_CO2
        self.current_ventilation_power = None
        self.current_time_interval = self.initial_time_step
        return self.current_state()

#implement CONTAM code here
    def _next_CO2(self, Ventilation_power):
        def Text_Generator():
            temporary_text = ""
            for i in range(25):
                temporary_text += str(i) + ":00:00 " + str(Actions["actions"][i]) + "\n"
            return temporary_text
        
        def CoSimulation():
            text = open("fatima_Main.prj").read()
            NextFile = open("fatima_second.prj","wt")
            NextFile.write(text[:3120] + Text_Generator() + text[3414:])
            NextFile.close()
            os.system("contamx3 fatima_second")
            os.system("simread fatima_second < responses.txt")
        
        def Dataset_Cleaning():   
            return pd.read_table("fatima_second.ncr")['1'] * 400/0.0006
        
        Actions = pd.DataFrame(range(25))
        Actions.columns = ["time"]
        Actions["actions"] = 0
        if Ventilation_power > 0:
            Actions["actions"][self.current_time_interval] = 1
        else: 
            Actions["actions"][self.current_time_interval] = 0
            
        New_state = Dataset_Cleaning()[self.current_time_interval +  self.__time_step_size ]
        plt.plot(Dataset_Cleaning())
        return New_state

# state : time_interval, current_int_temperature, current_ext_temperature
# state will be input to agent and agent will give amount of heating_cooling_power that will be needed to be applied
# heating_cooling_power will the action that will be taken

#this is the policy reward we want to control 
    def step(self, action):
        #Predicts by MODEL built in agent.
		#	Input : ventilation power
		#	Output :  Should return the Next CO2, Reward 
		
        self.current_ventilation_power = self.action_space[action]
        self.current_time_interval +=  self.__time_step_size
        if self.current_time_interval > 23:
            self.current_time_interval = 0
            

    	# calculate new CO2
        next_CO2 = self._next_CO2(self.current_ventilation_power)

    	# scaling reward factor for ventilation range penalty
		# Cost of provding ventilation
        operation_cost = self.current_ventilation_power
        L = 1
		# Cost of CO2 going outside the desired range of set points
        out_of_bounds_cost = abs(next_CO2 - self.CO2_setpoint)

		# As going out of bounds is more costly for us
		# total_cost = operation_cost - L * out_of_bounds_cost
        if self.CO2_setpoint <= next_CO2:
            total_cost = operation_cost + L * out_of_bounds_cost
        else:
            total_cost = L * out_of_bounds_cost - operation_cost

    		# Updating to new state
        self.current_int_CO2 = next_CO2

        return self.current_state(), total_cost
    

    def predicting_step(self):

		#Predicts by FORMULA the appropriate ventilaiton power to maintain 
		#the CO2 within setpoints

        def next_CO2(Ventilation_power):
            return self._next_CO2(Ventilation_power = Ventilation_power)
             
        next_CO2_no_power = next_CO2(0)

		# if CO2 is within limits by the system being off, do nothing 
        #return power as 0
        if (next_CO2_no_power < self.CO2_setpoint):
            return 0
		# Else, tehn ventilation is required 
        else:
            if next_CO2_no_power >= self.CO2_setpoint:
                return 1

    def current_state(self):

        return [self.current_time_interval, self.current_int_CO2]
    # In[45]:


# Creating Building environment which the agent will interact with
building = Building(maximum_ventilation_power = 1,time_step_size = 1, 
                 CO2_setpoint = 400, initial_time_step = 0, action_space = {0:0, 1:1})
# In[46]:

# Discount reward over time. Negative and positive rewards are handled seperately.
def discounted_reward(reward, i):
    return reward

# In[72]:
#initializing the state attributes, loading data and returning the first state
# the initial internal temperature of the biulding is passed as input
# equivalent to env.reset() from gym
initial_state = building.initialize_state(30)
#initial_state = 300
# Constants
n_actions = len(building.action_space)
input_shape = np.shape(initial_state)
intervals_per_day = 480
print(building.action_space)

# In[48]:


tf.reset_default_graph()
sess = tf.InteractiveSession()


# In[49]:


from keras.layers import Dense, InputLayer
from keras.models import Sequential

class Agent:
    
    def __init__(self, name, n_actions, input_shape,  epsilon, reuse = False):
        with tf.compat.v1.variable_scope(name, reuse):
            
            """
            Model Defination
            """
            
            self.network = Sequential()
            self.network.add(Dense(100, activation="relu", input_shape=input_shape ))
            self.network.add(Dense(100, activation="relu"))
            self.network.add(Dense(n_actions, activation="linear"))
            
            self.state_t = tf.placeholder("float32", [None, ] + list(input_shape))
            self.qvalues = self.get_symbolic_qvalues(self.state_t)
            
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        
    def get_qvalues(self, state_t):
        """
            Will be called in the function by the agent which will pass the state
            in the form of an array.
            Runs the graph that computes the qvalues
        """
        sess = tf.get_default_session()
        return sess.run(self.qvalues, {self.state_t : state_t})

    def get_symbolic_qvalues(self, state_t):
        """
        After get_qvalues function is called we get the 
        """

        qvalues = self.network(state_t)

        return qvalues

    def sample_actions(self, qvalues):
        """
            Sampling actions from the batch of qvalues
        """
        batch_size, n_actions = qvalues.shape

        #get random and best actions for each qvalue in batch
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        # choose whether to explore or exploit based on epsilon. Choice is made for each 
        #element in batch
        should_explore = np.random.choice([0,1], batch_size, p = [1-self.epsilon, self.epsilon])

        # Return Actions according to choice made in should explore

        return np.where(should_explore, random_actions, best_actions)


# In[70]:


agent = Agent("Agent", n_actions, input_shape, epsilon=0.5)
sess.run(tf.global_variables_initializer())


# In[51]:


target_network = Agent("target_network", n_actions, input_shape, epsilon=0)


# In[52]:


# Initializes/resets building

building.initialize_state(0)


# In[53]:


def restart_model(building, n_actions, input_shape, epsilon):
    """
    Called when we require to reset the agent with edits and restart the training
    from scratch
    """
    agent = Agent("Agent", n_actions, input_shape, epsilon=0.5)
    target_network = Agent("target_network", n_actions, input_shape, epsilon=0)
    building.initialize_state(0)
    sess.run(tf.global_variables_initializer())
    
    return agent, target_network


# In[54]:


# Evaluate function which will run our agent to work 
# Greedy mode will be run when we want to deploy the model and no longer run the model

def evaluate(agent, building, n_days, intervals_per_day, greedy = False):
    """
    Evaluates how well the model performs over a period of days.
    Returns the overall rewards
    """
    
    rewards = []
    for _ in range(n_days):
        reward = 0
        s = building.current_state()
        for i in range(intervals_per_day):
            
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r = building.step(action)
            #print(discounted_reward(r, i))
            reward += discounted_reward(r, i)
        
        rewards.append(reward)
        
    return np.mean(rewards)


# In[55]:


def play_and_record(agent, building, replay_buffer, n_iterations):
    """
    Run the agent to predict at "n_iterations" amount of time_intervals.
    Every step, the state, action, reward, next_state will be recorded in the 
    replay_buffer Record rewards over the time_intervals
    """
    
    s = building.current_state()
    
    reward = 0.0
    
    for t in range(n_iterations):
        
        qvalues = agent.get_qvalues([s])         # [s] cause we will pass s later in a batch
        a = agent.sample_actions(qvalues)[0]
        next_s, r = building.step(a)
        
        # Adding to replay buffer
        
        replay_buffer.add(state = s, action = a, reward = r, next_state = next_s)
        reward += r
        
    return reward


# In[56]:


def transfer_weights_to_target(agent, target_network):
    """
        During model designing we have defined self.weights that get weights of 
        the network with name.We train the agent network. Then after few 
        iterations we copy the weights from agent to target network

    """
    
    assigns = []
    
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        # Transfer weights layer by layer (i think so)
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
        
    # Do the abobe transfers per layer
    tf.get_default_session().run(assigns)


# In[57]:


# Creating placeholders for <s,a,r,s'> which will be fed by the replay buffer
# (None, )  is added cause we will add many observations of batchsize to train the network at once

states = tf.placeholder(tf.float32, shape = (None, ) + input_shape)
actions = tf.placeholder(tf.int32, shape = [None])
rewards = tf.placeholder(tf.float32, shape = [None])
next_states = tf.placeholder(tf.float32, shape = (None, ) + input_shape)

# gamma (how much to focus on future reward in calculating target Q(s,a))

gamma = 0.9


# In[58]:


"""
get qvalues Q(s) from state. Since we're already dealing with tensors we dont call agent.get_qvalues 
which was used earlier.

We get Q(s,a) by multiplying a one hot vector of actions with the current_qvalues. This is the first
component of the loss function which is computed by the agent network. This is the approximated/predicted Q(s,a)
"""
current_qvalues = agent.get_symbolic_qvalues(states)

current_action_qvalues = tf.reduce_sum(tf.one_hot(actions, n_actions) * current_qvalues, axis=1)


# In[59]:


"""
The second part is the target Q(s,a) value which we get by reward. We calculate Q(next_state) by using the
target network which has older weights than target
"""

next_qvalues = target_network.get_symbolic_qvalues(next_states)

next_state_value = tf.reduce_max(next_qvalues, axis=-1)

# reference/target qvalue
# We divide rewards by a factor p, also clip values to max -1
#p = 100

target_action_qvalues = rewards + gamma * next_state_value


# In[60]:


# calculating loss

td_loss = tf.reduce_mean((current_action_qvalues - target_action_qvalues)**2)

# Learning rate at 0.001. We use adam optimizer and train on weights on agent

train_step = tf.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=agent.weights)


# In[61]:


sess.run(tf.global_variables_initializer())


# In[62]:


# function to assign values to the placeholders computing loss from the experience replay buffer

def sample_batch(exp_replay, batch_size):
    sample = exp_replay.sample(batch_size)
    states_batch, action_batch, reward_batch, next_states_batch = sample["state"],sample["action"], sample["reward"],sample["next_state"]
    
    # temporary fix of shape which required (10, ) but got (10,1)
    action_batch = action_batch.ravel()
    reward_batch = reward_batch.ravel()
    
    return {
        states : states_batch,
        actions : action_batch,
        rewards : reward_batch,
        next_states : next_states_batch
    }


# In[63]:


from tqdm import trange
from IPython.display import clear_output
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


agent, target_network = restart_model(building, n_actions, input_shape, 0.5)


# In[65]:


get_ipython().system('pip install cpprb')
 


# In[66]:


from cpprb import ReplayBuffer

mean_rw_history = []
td_loss_history = []

buffer_size = 20000
state_shape = 2
act_dim = 1
reward_dim = 1


exp_replay = ReplayBuffer(20000, env_dict = {"state": {"shape": state_shape}, "action": {"shape": act_dim}, 
                                           "reward": {"shape": reward_dim}, "next_state": {"shape": state_shape} })
play_and_record(agent, building, exp_replay, 10_000)


# In[ ]:


for i in range(100_000):
    
    # Run the agent
    
    play_and_record(agent, building, exp_replay, 100)
    
    # train the network with batch size 64
    
    _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, 64))
    td_loss_history.append(loss_t)
    
    # We adjust epsion of agent and transfer weights from agent to target_network
    # Also evaluate agents performance in one day
    if i % 500 == 0:
        
        transfer_weights_to_target(agent, target_network)
        
        agent.epsilon = max(0.01, agent.epsilon * 0.99)
        mean_rw_history.append(evaluate(agent=agent, building=building, n_days=1, intervals_per_day=intervals_per_day))
    if i == 200:
        td_loss_history = []
        
    if i % 100 == 0:
        clear_output(True)
        print(f'iteration {i} and epsilon {agent.epsilon}')
        
        plt.figure(figsize=[48, 4])
        plt.subplot(1,2,1)
        plt.title("mean reward for one day")
        plt.plot(mean_rw_history)
        plt.grid()
        
        plt.figure(figsize=[48, 4])
        plt.subplot(1,2,2)
        plt.title("TD loss")
        plt.plot(td_loss_history)
        plt.grid()
        
        plt.show()
        


# In[ ]:


agent.network.save("trained_weights.h5")


# In[ ]:


def evaluate_by_algo(building, n_days, intervals_per_day, greedy = False):
    
    rewards = []
    for _ in range(n_days):
        reward = 0
        s = building.current_state()
        for i in range(intervals_per_day):

            action = building.predicting_step()
            s, r = building.step(action)
            reward += discounted_reward(r, i)
        
        rewards.append(reward)
        
    return rewards


# In[ ]:


def evaluate_by_model(agent, building, n_days, intervals_per_day, greedy = False):
    
    rewards = []
    for _ in range(n_days):
        reward = 0
        s = building.current_state()
        for i in range(intervals_per_day):
            
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0]
            s, r = building.step(action)
            reward += discounted_reward(r, i)
        
        rewards.append(reward)
        
    return rewards



# In[ ]:


agent.network.load_weights("trained_weights.h5")


# In[ ]:


building.initialize_state(22)
rewards_model = evaluate_by_model(agent=agent, building=building, n_days=56, intervals_per_day=intervals_per_day)
building.initialize_state(22)
rewards_algo = evaluate_by_algo(building, 56, intervals_per_day)


# In[ ]:


plt.plot(range(56), rewards_model, label = "model")
plt.plot(range(56), rewards_algo, label= "algo")
plt.legend()
plt.show()
