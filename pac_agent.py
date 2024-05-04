from collections import defaultdict
import numpy as np

class Agent(object):
    def __init__(
            self, action_space,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon_greedy=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.95,
            ):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Define the q_table
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        

    def choose_action(self, state, testing=False):
        state = self._convert_to_hashable(state)

        # do a random move, less frequent with lower epsilon
        if np.random.random() < self.epsilon and not testing:
            action = self.action_space.sample()

        # do the best move
        else:
            q_vals = self.q_table[state]
            
            perm_actions = np.random.permutation(8)
            q_vals = [q_vals[a] for a in perm_actions]

            perm_q_argmax = np.argmax(q_vals)
            action = perm_actions[perm_q_argmax]

        return action

    # game state needs to be hashable to pass as a key
    # recursively sets all elements of np array into hashable tuples
    def _convert_to_hashable(self, obj, visited=None):
        if visited is None:
            visited = set()

        if id(obj) in visited:
            # If the object has already been visited, return a unique placeholder value
            return str(id(obj))

        visited.add(id(obj))

        if isinstance(obj, np.ndarray) or isinstance(obj, list):
            return tuple(self._convert_to_hashable(item, visited) for item in obj)
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_hashable(item, visited) for item in obj)
        else:
            return obj  # Return unchanged for other types

    def _learn(self, info, obs, action, reward, finished, next_obs):
        
        # hash game states
        obs = self._convert_to_hashable(obs)
        next_obs = self._convert_to_hashable(next_obs)

        q_val = self.q_table[obs][action]

        #pacman died
        if reward == -1:
            q_target = reward - 1000
        #ate small pellet
        elif reward % 10 and reward < 50:
            q_target = reward + 10
        #ate big pellet 
        elif reward % 50 and reward < 100:
            q_target = reward + 75
        #did really good (ate a ghost or fruit)
        elif reward % 100 == 0:
            q_target = reward + 100
        
        #punish pacman for not eating pellets
        else:
            q_target = reward - 10
        
        # Update the q_table
        self.q_table[obs][action] += self.lr * (q_target - q_val)

        # Adjust the epsilon
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay