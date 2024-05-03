import numpy as np
from collections import defaultdict

'''
    This agent definition is adapted from the OpenAI Gym tutorial, 
    https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/,
    with some variable name changes and a slight modification to allow for this agent
    to be in a separate file (blackjack_training.py) from where the training takes place.
'''

class Q_Agent:
    def __init__(self, action_space, learning_rate: float, initial_e: float, decay_e: float, final_e: float, discount_factor: float):
        self.action_space = action_space
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_e
        self.decay_epsilon = decay_e
        self.final_epsilon = final_e
        self.training_error = []
    
    # returns the best action with P(1 - initial_epsilon)
    # otherwise a random actiion with P(initial_epsilon) to ensure exploration
    def get_action(self, observation: tuple[int, int, bool]) -> int:
        if np.random.random() < self.initial_epsilon:
            # P(initial_epsilon) returns a random action from the action space
            return self.action_space.sample()
        else:
            # P(1 - initial_epsilon)
            return int(np.argmax(self.q_table[observation]))

    def update(self, observation: tuple[int, int, bool], action: int, reward: float, finished: bool, next_observation: tuple[int, int, bool]):
        # update the q table
        new_q_table = (not finished) * np.max(self.q_table[next_observation])

        temporal_diff = reward + self.discount_factor * new_q_table - self.q_table[observation][action]
        self.q_table[observation][action] = self.q_table[observation][action] + self.learning_rate * temporal_diff

        self.training_error.append(temporal_diff)

    def epsilon_decay(self):
        self.initial_epsilon = max(self.final_epsilon, self.initial_epsilon - self.decay_epsilon)