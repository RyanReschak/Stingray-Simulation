from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.losses import MeanSquaredError
from collections import deque
import numpy as np
import random

class DQL:
    def __init__(self, num_states, num_actions, l_rate=0.9, dis_rate=0.9, epsilon=1, decay=0.985, target_update_rate=3):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.max_replay_len = 2500
        self.replay_memory = deque(maxlen=self.max_replay_len)

        self.discount_rate = dis_rate
        self.learning_rate = l_rate

        self.epsilon = epsilon
        self.eps_decay = decay
        self.eps_min = 0.05

        self.model = Sequential([
            Dense(30, activation='relu', input_dim=self.num_states),
            Dropout(.05),
            Dense(30, activation='relu'),
            Dropout(0.1),
            Dense(30, activation='relu', input_dim=self.num_states),
            Dropout(.05),
            #output layer
            Dense(self.num_actions, activation='linear')
            ])
        self.model.compile(loss='msle',#MeanSquaredError(reduction="auto"),
                           optimizer=Adam(lr=self.learning_rate),
                           metrics=['accuracy'])
        
        #Clone Policy Network
        self.target_network = self.model
        self.target_updatet_rate = target_update_rate
    
    def update_replay_memory(self, state, action, reward, next_state):
        self.replay_memory.append([state, action, reward, next_state])
    
    def predict_action(self, state):
        return np.argmax(self.model.predict(state)[0])
    
    def action_choice(self, state):
        n = np.random.random_sample()
        if (n > self.epsilon):
            #print(self.model.predict(state))
            #print(state)
            return self.predict_action(state) 
        return np.random.randint(low=0,high=self.num_actions)
    
    def replay(self, batch_size, episode):
        
        #Select the trianing data
        select_batch = random.sample(self.replay_memory, batch_size)
        #delete the training data so that it isn't reused
        for i in range(batch_size):
            self.replay_memory.pop() 

        for state, action, reward, next_state in select_batch:
            #Make target_NN be copy of model and then use that in replay here
            #just for testing right now
            target = reward + self.discount_rate * np.argmax(self.target_network.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][:] = 0
            #print(state, action,reward)
            target_f[0][action] = target
            #print(target_f)

            self.model.fit(state, target_f, verbose=0)
        
        #print(target_f)
        
        if episode % self.target_updatet_rate == 0:
            self.target_network = self.model
            if self.epsilon > self.eps_min:
                self.epsilon *= self.eps_decay

    def load(self, file_location):
        self.model.load_weights(file_location)

    def save(self, file_location):
        self.model.save_weights(file_location)
