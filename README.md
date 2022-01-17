# Policy Gradient Algorithms From Scratch (NumPy)
This repository showcases two policy gradient algorithms (One Step [Actor Critic](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) and [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)) applied to two MDPs. The algorithms are implemented from scratch with Numpy and utilize linear regression for the value function and single layer Softmax for the policy. The MDPs are: Gridworld and Mountain Car.




## Run Instructions
### Packages:
numpy and matplotlib

Create virtual environment, install requirements and run:
(windows instructions)
1. Run `python -m venv venv`
1. Run `.\venv\Scripts\activate` (windows)
1. Run `pip install -r requirements.txt`
1. Run `python .\experiments.py`
be wary of long compute times and plots that will pop up and must be exited in order to comtinue.

## Some Sample Plots
<p float="left">
    <img src="./plots/g2%20aoe%20plt=4%20mountain_car%20alg=ac%20O=many%20v_alpha=0.0001_%20p_alpha=0.001.png" width="275">
    <img src="./plots/g6%20aoe%20plt=1%20mountain_car%20alg=ac%20O=20%20v_alpha=0.001_%20p_alpha=0.01.png" width="275">
    <img src="./plots/g7%20aoe%20plt=2%20mountain_car%20alg=both%20O=many%20v_alpha=x_%20p_alpha=x.png" width="275">
</p>

## Files
* `experiments.py` - Runs pre programmed experiments that output various plots both in the terminal and saved to .png files.
* `mdp.py` - Contains two MDP domains: Gridworld and Mountain Car, that the experiments are run on.
* `models.py` - Contains ValueFunction and Policy which are the two models used (linear layers) for function approximation by the algorithms.
* `policy_gradient_algorithms.py` - Contains the policy gradient algorithms One Step Actor Critic and Proximal Policy Optimization (PPO).

[MIT License](/license)