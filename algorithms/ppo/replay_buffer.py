import numpy as np
from collections import deque
import random

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, log_prob, reward, next_state):
        experience = (state, action, log_prob, reward, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        log_prob_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, log_prob, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            log_prob_batch.append(log_prob)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
        
        return state_batch, action_batch, log_prob_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)