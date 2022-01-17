
import numpy as np
import matplotlib.pyplot as plt

from mdp import Gridworld, MountainCar
from policy_gradient_algorithms import one_step_actor_critic, proximal_policy_optimization



class Experiments:

    gridworld_final_performance_episodes = 20
    gridworld_actions_compare_value = 8
    gridworld_y_min = 0
    gridworld_y_max = 120
    gridworld_episodes = 100
    gridworld_max_actions = 100
    gridworld_discount = 0.9

    mountain_car_final_performance_episodes = 50
    mountain_car_actions_compare_value = 90
    mountain_car_y_min = 60
    mountain_car_y_max = 700
    mountain_car_episodes = 500
    mountain_car_max_actions = 650
    mountain_car_discount = 1.0
    print_decimal = "{:.3f}"

    ac_alg="ac"
    ppo_alg="ppo"

    def __init__(self):
        pass


    def run(self):
        self.actor_critic_gridworld()
        self.ppo_gridworld()
        self.actor_critic_mountain_car()
        self.ppo_mountain_car()
        self.ac_vs_ppo()

    # helper graph function
    def graph_actions_over_episodes(self, graph_name, means, stds, mdp, alg, policy_alpha, value_function_alpha, actions_compare_value, min, max, metrics, plot_std, comparison_metric = "Order ", labels = None):

        plt.clf()
        plots = len(means)
        colors = ["green", "blue", "red", "purple", "cyan", "yellow"]
        x_range = np.arange(means[0].shape[0])
        plt.ylim(min, max)
        plt.axhline(y=actions_compare_value, color='red', linestyle='-', alpha=0.5)

        if plots == 1:
            metrics = [""]

        for i in range(plots):
            if labels == None:
                plt.plot(x_range, means[i], color=colors[i], label=comparison_metric+ str(metrics[i]))
            else:
                plt.plot(x_range, means[i], color=colors[i], label=labels[i])
            
            if plot_std:
                plt.fill_between(x_range, means[i] + 2*stds[i], means[i] - 2*stds[i], color=colors[i], alpha=0.3)
        
        if plots > 1:
            order = "many"
            plt.legend(loc='upper right')
        else:
            order = str(mdp.complexity)

        plt.xlabel("Episode")
        plt.ylabel("Actions Taken Per Episode")
        plt.title("Actions Per Episode")

        name = graph_name + "aoe" + " plt=" + str(plots) + " " + mdp.name + " alg=" + alg + " O=" + order + " v_alpha=" + str(value_function_alpha) + "_ p_alpha=" + str(policy_alpha) + ".png"
        plt.savefig("./plots/" + name)
        plt.show()



    # outputs data for a table and a graph
    def actor_critic_gridworld(self):
        print("\nActor Critic on Gridworld")
        iterations = 25
        v_alphas = [.05, 0.1, 0.25, 0.5, 1.0]
        p_alphas = [.05, 0.1, 0.25, 0.5, 1.0]
        best_v_alpha = None
        best_p_alpha = None
        best_performance_mean = 100000000
        best_performance_std = None
        print("Value Alpha | Policy Alpha | Performance Mean | Performance Std")
        for v_alpha in v_alphas:
            for p_alpha in p_alphas:
                gridworld = Gridworld(self.gridworld_discount)
                actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std = one_step_actor_critic(gridworld, v_alpha, p_alpha, iterations, self.gridworld_episodes, self.gridworld_max_actions, self.gridworld_final_performance_episodes)
                print(v_alpha, p_alpha, self.print_decimal.format(final_performance_mean), self.print_decimal.format(final_performance_std))
                
                if final_performance_mean < best_performance_mean:
                    best_performance_mean = final_performance_mean
                    best_performance_std  = final_performance_std
                    best_v_alpha = v_alpha
                    best_p_alpha = p_alpha

        print("The best learning rates found were value alpha = " + str(best_v_alpha) +" and policy alpha = " + str(best_p_alpha) + " with a mean performance = " + str(best_performance_mean) + " and std = " + str(best_performance_std))

        # graph best one
        iterations = 50
        gridworld = Gridworld(self.gridworld_discount)
        actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std = one_step_actor_critic(gridworld, best_v_alpha, best_p_alpha, iterations, self.gridworld_episodes, self.gridworld_max_actions, self.gridworld_final_performance_episodes)
        self.graph_actions_over_episodes("g1 ", [actions_taken_average], [actions_taken_std], gridworld, self.ac_alg, best_p_alpha, best_v_alpha, self.gridworld_actions_compare_value, self.gridworld_y_min, self.gridworld_y_max, None, True)




    def actor_critic_mountain_car(self):
        print("\nActor Critic on Mountain Car")
        
        # find best order using special n-plot order graph
        iterations = 30
        orders = [10, 13, 17, 20]
        v_alpha = 0.0001
        p_alpha = 0.001
        self.mountain_car_actions_over_episodes_fourier_complexity(orders, v_alpha, p_alpha, iterations, False)
        

    def mountain_car_actions_over_episodes_fourier_complexity(self, orders, value_function_alpha, policy_alpha, iterations, show_stds):
        print("Iterating over orders...")
        order_means = []
        order_stds = []
        best_performance_mean = 100000000
        best_performance_std = None
        best_order_means = None
        best_order_stds = None
        for order in orders:
            mountain_car = MountainCar(self.mountain_car_discount, "fourier", order)

            actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std = one_step_actor_critic(mountain_car, value_function_alpha, policy_alpha, iterations, self.mountain_car_episodes, self.mountain_car_max_actions, self.mountain_car_final_performance_episodes)

            if final_performance_mean < best_performance_mean:
                best_performance_mean = final_performance_mean
                best_performance_std  = final_performance_std
                best_order = order
                best_order_means = actions_taken_average
                best_order_stds = actions_taken_std

            print("order: ", order, "final performance mean: ", final_performance_mean, "final performance std: ", final_performance_std)
            order_means.append(actions_taken_average)
            order_stds.append(actions_taken_std)

        # graph all orders
        self.graph_actions_over_episodes("g2 ", order_means, order_stds, mountain_car, self.ac_alg, policy_alpha, value_function_alpha, self.mountain_car_actions_compare_value, self.mountain_car_y_min, self.mountain_car_y_max, orders, show_stds)

        # graph best order with it's std
        mountain_car = MountainCar(self.mountain_car_discount, "fourier", best_order)
        print("best model: order: ", best_order, "final performance mean: ", best_performance_mean, "final performance std: ", best_performance_std)
        self.graph_actions_over_episodes("g3 ", [best_order_means], [best_order_stds], mountain_car, self.ac_alg, policy_alpha, value_function_alpha, self.mountain_car_actions_compare_value, self.mountain_car_y_min, self.mountain_car_y_max, None, True)



    # outputs data for a table and a graph
    def ppo_gridworld(self):
        print("\nProximal Policy Optimization on Gridworld")
        
        # PPO specific hyperparameters
        rollout_episodes = 1
        clip = 0.2
        epochs = 20
    
        iterations = 25
        v_alphas = [.05, 0.1, 0.25, 0.5, 0.75]
        p_alphas = [.05, 0.1, 0.25, 0.5, 0.75]
        best_v_alpha = None
        best_p_alpha = None
        best_performance_mean = 100000000
        best_performance_std = None
        gridworld = Gridworld(self.gridworld_discount)

        print("Value Alpha | Policy Alpha | Performance Mean | Performance Std")
        for v_alpha in v_alphas:
            for p_alpha in p_alphas:
                actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std = proximal_policy_optimization(gridworld, v_alpha, p_alpha, clip, iterations, self.gridworld_episodes, self.gridworld_max_actions, rollout_episodes, epochs, self.gridworld_final_performance_episodes)

                print(v_alpha, p_alpha, self.print_decimal.format(final_performance_mean), self.print_decimal.format(final_performance_std))
                
                if final_performance_mean < best_performance_mean:
                    best_performance_mean = final_performance_mean
                    best_performance_std  = final_performance_std
                    best_v_alpha = v_alpha
                    best_p_alpha = p_alpha

        print("The best learning rates found were value alpha = " + str(best_v_alpha) +" and policy alpha = " + str(best_p_alpha) + " with a mean performance = " + str(best_performance_mean) + " and std = " + str(best_performance_std))

        # graph best one
        iterations = 50
        gridworld = Gridworld(self.gridworld_discount)
        actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std = proximal_policy_optimization(gridworld, best_v_alpha, best_p_alpha, clip, iterations, self.gridworld_episodes, self.gridworld_max_actions, rollout_episodes, epochs, self.gridworld_final_performance_episodes)
        
        self.graph_actions_over_episodes("g4 ", [actions_taken_average], [actions_taken_std], gridworld, self.ppo_alg, best_p_alpha, best_v_alpha, self.gridworld_actions_compare_value, self.gridworld_y_min, self.gridworld_y_max, None, True)
    



    def ppo_mountain_car(self):
        print("\nPPO on Mountain Car")
        # PPO specific hyperparameters: (all fixed at these values)
        clip = 0.2
        rollout_episodes = 1
        epochs = [5,8,10,12]

        order = 20
        v_alpha = 0.001
        p_alpha = 0.01
        iterations = 20

        epoch_means = []
        epoch_stds = []
        best_performance_mean = 100000000
        best_performance_std = None
        best_epoch_means = None
        best_epoch_stds = None
        best_epoch = None

        mountain_car = MountainCar(self.mountain_car_discount, "fourier", order)

        print("Iterating over epochs...")
        for epoch in epochs:
            actions_taken_average, actions_taken_std, final_performance_mean, final_performance_std = proximal_policy_optimization(mountain_car, v_alpha, p_alpha, clip, iterations, self.mountain_car_episodes, self.mountain_car_max_actions, rollout_episodes, epoch, self.mountain_car_final_performance_episodes)

            if final_performance_mean < best_performance_mean:
                best_performance_mean = final_performance_mean
                best_performance_std  = final_performance_std
                best_epoch = epoch
                best_epoch_means = actions_taken_average
                best_epoch_stds = actions_taken_std

            print("epochs: ", epoch, "final performance mean: ", final_performance_mean, "final performance std: ", final_performance_std)
            epoch_means.append(actions_taken_average)
            epoch_stds.append(actions_taken_std)

        # graph all orders
        self.graph_actions_over_episodes("g5 ", epoch_means, epoch_stds, mountain_car, self.ac_alg, p_alpha, v_alpha, self.mountain_car_actions_compare_value, self.mountain_car_y_min, self.mountain_car_y_max, epochs, False, "Learning Epochs: ")

        # graph best order with it's std
        print("best model: learning epochs: ", best_epoch, "final performance mean: ", best_performance_mean, "final performance std: ", best_performance_std)

        self.graph_actions_over_episodes("g6 ", [best_epoch_means], [best_epoch_stds], mountain_car, self.ac_alg, p_alpha, v_alpha, self.mountain_car_actions_compare_value, self.mountain_car_y_min, self.mountain_car_y_max, None, True)



    def ac_vs_ppo(self):
        print("\nComparing AC vs PPO...")

        # Mountain car
        print("Evaluation on Mountain Car...")
        clip = 0.2
        rollout_episodes = 1
        order = 20

        iterations = 25

        mountain_car = MountainCar(self.mountain_car_discount, "fourier", order)

        actions_taken_average_ac, actions_taken_std_ac, final_performance_mean_ac, final_performance_std_ac = one_step_actor_critic(mountain_car, 0.0001, 0.001, iterations, self.mountain_car_episodes, self.mountain_car_max_actions, self.mountain_car_final_performance_episodes)

        actions_taken_average_ppo, actions_taken_std_ppo, final_performance_mean_ppo, final_performance_std_ppo = proximal_policy_optimization(mountain_car, 0.001, 0.01, clip, iterations, self.mountain_car_episodes, self.mountain_car_max_actions, rollout_episodes, 12, self.mountain_car_final_performance_episodes)

        # Compare results
        print("Actor Critic Final Performance Mean:", final_performance_mean_ac, " final performance std: ", final_performance_std_ac)
        print("PPO Final Performance Mean:", final_performance_mean_ppo, " final performance std: ", final_performance_std_ppo)

        # graph both side by side
        self.graph_actions_over_episodes("g7 ", [actions_taken_average_ac, actions_taken_average_ppo], [actions_taken_std_ac, actions_taken_std_ppo], mountain_car, "both", "x", "x", self.mountain_car_actions_compare_value, self.mountain_car_y_min, self.mountain_car_y_max, None, False, None, ["Actor Critic", "PPO"])
        

        # Mountain Car
        print("Evaluation on Gridworld...")
        clip = 0.2
        rollout_episodes = 1

        iterations = 50

        gridworld = Gridworld(self.gridworld_discount)

        actions_taken_average_ac, actions_taken_std_ac, final_performance_mean_ac, final_performance_std_ac = one_step_actor_critic(gridworld, 0.25, 0.5, iterations, self.gridworld_episodes, self.gridworld_max_actions, self.gridworld_final_performance_episodes)

        actions_taken_average_ppo, actions_taken_std_ppo, final_performance_mean_ppo, final_performance_std_ppo = proximal_policy_optimization(gridworld, 0.25, 0.75, clip, iterations, self.gridworld_episodes, self.gridworld_max_actions, rollout_episodes, 20, self.gridworld_final_performance_episodes)

        # Compare results
        print("Actor Critic Final Performance Mean:", final_performance_mean_ac, " final performance std: ", final_performance_std_ac)
        print("PPO Final Performance Mean:", final_performance_mean_ppo, " final performance std: ", final_performance_std_ppo)

        # graph both side by side
        self.graph_actions_over_episodes("g8 ", [actions_taken_average_ac, actions_taken_average_ppo], [actions_taken_std_ac, actions_taken_std_ppo], gridworld, "both", "x", "x", self.gridworld_actions_compare_value, self.gridworld_y_min, self.gridworld_y_max, None, False, None, ["Actor Critic", "PPO"])


def main():
    experiments = Experiments()
    experiments.run()


if __name__ == "__main__":
    main()