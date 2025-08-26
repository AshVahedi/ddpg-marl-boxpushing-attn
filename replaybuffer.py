import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return {"states":states,"actions": actions, "rewards": rewards, "next_states":next_states,"dones": dones}
    def delete(self):
        """Clears the entire buffer."""
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
