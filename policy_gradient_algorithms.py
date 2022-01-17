import numpy as np
from models import Policy, ValueFunction


def one_step_actor_critic(mdp, value_function_alpha, policy_alpha, iterations, episodes, max_actions, final_performance_episodes):
    #print("One Step actor critic with mdp: " + mdp.name)

    actions_vs_episodes_all = np.zeros((iterations, episodes))
    for i in range(iterations):

        policy = Policy(policy_alpha, mdp.state_features_length, mdp.actions_length)
        value_function = ValueFunction(value_function_alpha, mdp.state_features_length)

        #print("Iteration: " + str(i))
        actions_vs_episodes = []
        for episode in range(episodes):

            state = mdp.get_start_state()
            state_features = mdp.get_state_features(state)
            #print(state_features)
            state_value = value_function.evaluate_state(state_features)
            actions = 0

            while not mdp.episode_over(state):
                actions += 1
                if actions == max_actions:
                    break
                # draw action from policy
                action, action_probabilities = policy.get_action(state_features, True)

                # execute action a, observe r and s'
                next_state = mdp.get_next_state(state, action)
                reward = mdp.get_reward(next_state)

                # get state features from state
                next_state_features = mdp.get_state_features(next_state)

                # compute TD error
                next_state_value = value_function.evaluate_state(next_state_features)
                target = reward + mdp.discount * next_state_value
                td_error = target - state_value

                # update actor & critic
                policy.update_parameters(td_error, state_features)
                value_function.update_parameters(td_error, state_features)

                # s = s'
                state = next_state
                state_features = next_state_features
                state_value = next_state_value

        
            actions_vs_episodes.append(actions)
        actions_vs_episodes_all[i] = np.array(actions_vs_episodes)

    return collect_statistics(actions_vs_episodes_all, final_performance_episodes)



# ppo works by running the environment for some steps and then computing stochastic gradient descent on the collected s,a,r,s' examples
# I use a value function for the baseline just like in actor-critic
def proximal_policy_optimization(mdp, value_function_alpha, policy_alpha, clip, iterations, episodes, max_actions, rollout_episodes, epochs, final_performance_episodes):
    #print("Proximal Policy Optimization with mdp: " + mdp.name)
    actions_vs_episodes_all = np.zeros((iterations, episodes))

    for i in range(iterations):
        #print("Iteration: " + str(i))
        policy = Policy(policy_alpha, mdp.state_features_length, mdp.actions_length, clip)
        value_function = ValueFunction(value_function_alpha, mdp.state_features_length)
        actions_vs_episodes = []

        episode = 0
        while episode < episodes:
            current_rollout_episodes = min(episodes - episode, rollout_episodes)
            # run the environment
            state_features, actions, action_probabilities, targets, advantages, episode_lengths = ppo_rollout(mdp, policy, value_function, current_rollout_episodes, max_actions)

            episode += rollout_episodes
            actions_vs_episodes += episode_lengths

            # SGD on policy and value function
            for j in range(epochs):
                policy.ppo_gradient_step(state_features, action_probabilities, actions, advantages)
                
                state_values = value_function.evaluate_state(state_features)
                errors = targets - state_values
                value_function.update_parameters(errors, state_features)
                
        # going to take average over all ppo iterations
        actions_vs_episodes_all[i] = np.array(actions_vs_episodes)

    return collect_statistics(actions_vs_episodes_all, final_performance_episodes)



def ppo_rollout(mdp, policy, value_function, rollout_episodes, max_actions):

    state_features = []
    actions = []
    action_probabilities = []
    rewards = []

    last_state_features = []
    episode_lengths = []

    # collect rollout_episodes trajectories
    for ep in range(rollout_episodes):
        state = mdp.get_start_state()
        state_feature = mdp.get_state_features(state)
        actions_taken = 0

        while not mdp.episode_over(state):
            if actions_taken == max_actions:
                break
            # draw action from policy
            action, action_probability = policy.get_action(state_feature, False)

            # execute action a, observe r and s'
            next_state = mdp.get_next_state(state, action)
            reward = mdp.get_reward(next_state)

            # record for stachastic gradient descent later
            state_features.append(state_feature[0])
            actions.append(action)
            action_probabilities.append(action_probability)
            rewards.append(reward)

            # s = s'
            state = next_state
            state_feature = mdp.get_state_features(state)

            actions_taken += 1

        last_state_features.append(state_feature) # v(s') for the last example in this episode batch (need for Advantage function)
        episode_lengths.append(actions_taken)

    state_features = np.array(state_features)
    actions = np.array(actions)
    action_probabilities = np.array(action_probabilities)
    rewards = np.array(rewards)

    # make sure state_values and next_state_values line up
    ep_begin = 0
    state_values = np.squeeze(value_function.evaluate_state(state_features))
    last_state_values = np.reshape(np.squeeze(value_function.evaluate_state(last_state_features)), (-1))
    next_state_values = np.zeros(state_values.shape)
    # this makes sure the state_value array and next_next_value arrays are lined up so that the td errors can easily be computed
    for i in range(rollout_episodes):
        l = episode_lengths[i]
        next_state_values[ep_begin: ep_begin+l-1] = state_values[ep_begin+1:ep_begin+l]
        next_state_values[ep_begin+l-1] = last_state_values[i]
        ep_begin += l

    targets = rewards + mdp.discount * next_state_values # super easy td error calculation
    advantages = targets - state_values

    return (state_features, actions, action_probabilities, targets, advantages, episode_lengths) 



# basic statistics for my plots
def collect_statistics(actions_vs_episodes_all, final_performance_episodes):
    actions_taken_average = np.mean(actions_vs_episodes_all, 0)
    actions_taken_std = np.std(actions_vs_episodes_all, 0)

    final_performance_mean = np.mean(actions_taken_average[-final_performance_episodes:])
    final_performance_std = np.mean(actions_taken_std[-final_performance_episodes:])

    return actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std
