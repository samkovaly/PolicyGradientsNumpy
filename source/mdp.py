
import math
import random
import numpy as np

# helper function
def rand_tuple(lower, higher):
    y = random.randint(lower, higher)
    x = random.randint(lower, higher)
    return (y,x)

# I reused most of my MDP code on grildworld from previous homeworks
class Gridworld:
    name = "gridworld"
    state_size_y = 5
    state_size_x = 5
    start_state = (0,0)
    water_state = (4,2)
    goal_state = (4,4)
    obstacles = [(2,2),(3,2)]
    
    goal_reward = 10
    water_reward = -10
    
    actions = [
        (0, "\u2192"),
        (math.pi/2, "\u2191"),
        (math.pi, "\u2190"),
        (-math.pi/2, "\u2193")
    ]
    actions_length = 4

    # p(s,a,s') - (veer angle, Probability)
    veer_transitions = [(0, 0.8), (-1, 0.1), (math.pi/2, 0.05), (-math.pi/2, 0.05)]

    def __init__(self, discount):
        self.discount = discount
        self.state_features_length = self.state_size_y * self.state_size_x
        self.complexity = None


    def get_start_state(self):
        return self.start_state

    def get_next_state(self, state, action):
        action_angle = self.actions[action][0]

        transition_probability = random.random()
        prob_sum = 0
        for veer_transition in self.veer_transitions:
            prob_sum += veer_transition[1]
            if transition_probability < prob_sum:
                if(veer_transition[0] == -1):
                    next_state = state
                else:
                    next_state = self.get_next_position(state, action_angle, veer_transition[0])
                return next_state

    def get_next_position(self, state, action_angle, veer_angle):
        angle = action_angle + veer_angle
        direction = (int(math.sin(angle) * -1), int(math.cos(angle)))
        next_state = (state[0] + direction[0], state[1] + direction[1])

        # boundaries
        if(next_state[0] < 0 or next_state[0] >= self.state_size_y or next_state[1] < 0 or next_state[1] >= self.state_size_x) \
            or next_state in self.obstacles:
            return state
        else:
            return next_state

    # gridworld reward only depends on the next state we are entering
    def get_reward(self, next_state):
        if(next_state == self.goal_state):
            return self.goal_reward
        if(next_state == self.water_state):
            return self.water_reward
        return 0
        
    def episode_over(self, state):
        return state == self.goal_state

    def get_state_features(self, state):
        state = np.reshape(state, (-1,2))
        y = state[:,0]
        x = state[:,1]
        # return a 1d vector of length width*height with 1 at the agent's location and 0 elsewhere
        features = np.zeros( (state.shape[0], self.state_size_y * self.state_size_x ) )
        features[:, y * self.state_size_y + x] = 1
        return features




# I reused most of my MDP code on mountain_car from previous homeworks
class MountainCar:
    name = "mountain_car"
    all_reward = -1
    goal_reward = 0
    x_max = 0.5
    x_min = -1.2
    v_max = 0.7
    v_min = -0.7

    actions = [
        (-1, "R"), # reverse
        (0, "N"), # neutral
        (1, "F")  # forward
    ]
    actions_length = 3
    
    def __init__(self, discount, feature_type, complexity):
        self.discount = discount
        self.feature_type = feature_type
        self.complexity = complexity
        if feature_type == "fourier":
            self.state_features_length = 1 + 2 * complexity

    def get_start_state(self):
        initial_x = random.uniform(-0.6, -0.4)
        return (initial_x, 0)

    def get_next_state(self, state, action):
        x = state[0]
        v = state[1]

        acceleration = self.actions[action][0]
        v_next = v + (0.001 * acceleration) - (0.0025 * math.cos(3 * x))
        x_next = x + v_next

        x_next = max(min(x_next, self.x_max), self.x_min)
        v_next = max(min(v_next, self.v_max), self.v_min)
        
        if x_next == self.x_min or x_next == self.x_max:
            v_next = 0

        return (x_next, v_next)

    def get_reward(self, next_state):
        if next_state[0] == self.x_max:
            return self.goal_reward
        else:
            return self.all_reward
        
    def episode_over(self, state):
        return state[0] == self.x_max

    def get_state_features(self, state):
        if self.feature_type == "fourier":
            return self.fourier_cos( np.reshape(state, (-1,2) ))
    
    def fourier_cos(self, state):
        x = state[:,0]
        v = state[:,1]
        x = np.reshape((x - self.x_min) / (self.x_max - self.x_min), (-1,1)) # 0 to 1 range
        v = np.reshape((v - self.v_min) / (self.v_max - self.v_min), (-1,1))
        #φ(s) = [1, cos(1πx), cos(2πx), . . . , cos(Mπx), cos(1πv), cos(2πv), . . . cos(Mπv)]>.
        fourier_x = np.cos(np.reshape(np.arange(self.complexity) + 1, (1,-1)) * np.pi * x)
        fourier_v = np.cos(np.reshape(np.arange(self.complexity) + 1, (1,-1)) * np.pi * v)
        fourier = np.concatenate( (np.ones((state.shape[0], 1)), fourier_x, fourier_v), 1 )
        return fourier