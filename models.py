import numpy as np
import warnings
warnings.filterwarnings("error") # for overflow

# differentiable parameterized policy
class Policy:
    parameters = None
    
    def __init__(self, alpha, state_features_length, actions_length, clip = None):
        self.alpha = alpha
        self.parameters = np.zeros((actions_length, state_features_length)) # So I don't have to keep transposing later
        self.clip = clip

    def get_action(self, state_features, compute_gradient):
        state_features = np.squeeze(state_features)
        # softmax over all actions - probability of taking them
        action_values = np.exp(np.matmul(self.parameters, state_features))
        normalization = np.sum(action_values, axis=0)
        action_probabilities = action_values / normalization

        # sample an action
        action_index = np.random.choice(np.arange(action_probabilities.shape[0]), 1, p=action_probabilities).item()

        if compute_gradient:
            # compute gradient based on rules derived in class
            state_features = np.reshape(state_features, (1,-1))
            action_probabilities = np.reshape(action_probabilities, (-1,1))

            ln_gradient = -1 * action_probabilities * state_features
            ln_gradient[action_index] = (1 - action_probabilities[action_index]) * state_features
            self.ln_gradient = ln_gradient

        return (action_index, np.squeeze(action_probabilities))


    def ppo_gradient_step(self, state_features, old_action_probabilities, actions, advantages):
        n_len = state_features.shape[0]
        s_len = state_features.shape[1]
        a_len = old_action_probabilities.shape[1]

        # get probability for every action at every state that was seen in the previous rollout
        state_features = np.expand_dims(np.transpose(state_features), 0)
        parameters = np.reshape(self.parameters, (a_len, s_len, 1))
        action_values = np.exp(np.sum(parameters * state_features, axis=1))
        normalization = np.sum(action_values, axis=0)
        current_action_probabilities = np.reshape(action_values / normalization, (a_len,1 , n_len))
        
        # compute ratio needed by the ppo algorithm
        old_action_probabilities = np.expand_dims(np.transpose(old_action_probabilities), 1)
        ratios = current_action_probabilities / old_action_probabilities

        gradients = (-1 * current_action_probabilities) * state_features # start with "not action" gradient first

        # shameful loop solution
        # both corrects the gradient for which actions were actually taken and zeros out gradients that violate our clip
        actions = np.squeeze(actions)
        for i in range(n_len):
            action = actions[i]
            gradients[action, :, i] = (1 - current_action_probabilities[action,0,i]) * state_features[0,:,i]
            for a in range(a_len):
                if not ((advantages[i] >= 0 and ratios[a,0,i] < 1 + self.clip) or (advantages[i] < 0 and ratios[a,0,i] > 1 - self.clip)):
                    gradients[a, :, i] = np.zeros(s_len)

        # compute final gradient. Notice that the zero gradients stay as zero no matter the advantage
        advantages = np.reshape(advantages, (1,1,n_len))
        gradients = gradients * advantages * ratios
        gradients = np.mean(gradients, axis=2)

        self.parameters += self.alpha * gradients
    

    def update_parameters(self, td_error, state_features):
        self.parameters += self.alpha * td_error * self.ln_gradient


# differentiable value function approximation
class ValueFunction:
    parameters = None
    
    def __init__(self, alpha, state_features_length):
        self.alpha = alpha
        self.parameters = np.zeros(state_features_length)

    def evaluate_state(self, state_features):
        return np.squeeze(np.matmul(state_features, np.reshape(self.parameters, (-1,1) )))

    def update_parameters(self, td_error, state_features):
        self.parameters += self.alpha * np.mean( np.reshape(td_error, (-1, 1)) * state_features, axis=0)