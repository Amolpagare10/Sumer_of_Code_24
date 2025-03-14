# 📚 Week 4 - Q-Learning and Solving MountainCar Environment

## 🚀 Overview

In Week 4, we shift our focus to **Q-Learning**, one of the fundamental algorithms in Reinforcement Learning.
Our goal is to "solve" the **MountainCar-v0** environment provided by **Gymnasium**, using discrete state-space
representation and Q-learning updates. This week provides a practical understanding of applying RL to real 
environments where the state and action spaces are continuous.

---

## 🎯 Objectives

- ✅ Understand Q-Learning and its update mechanism.
- ✅ Discretize continuous observation spaces for classical RL algorithms.
- ✅ Implement Q-Learning for MountainCar.
- ✅ Analyze agent performance through training metrics and visualization.

---

## 📦 File Structure
```
Week_4/
├── q_learning_mountain_car.py   # Main Q-learning implementation for MountainCar environment.
├── utils.py                    # Utility functions for state discretization, environment handling.
└── analysis_plots.py           # Scripts for plotting training performance, rewards, Q-tables, and learning curves.
```
---

## 🧠 Concepts Covered

- **Q-Learning**:
  - Off-policy Temporal Difference learning algorithm.
  - Learns optimal action-value function (Q-function) for state-action pairs.

- **Q-Learning Update Rule**:
  Q(s, a) ← Q(s, a) + α * [ r + γ * max_a' Q(s', a') - Q(s, a) ]

- **Discretization of Continuous Space**:
  - MountainCar observations (position, velocity) are continuous.
  - We discretize these into bins to create manageable state representations for Q-learning.

- **Exploration vs. Exploitation**:
  - ε-greedy policy used to balance exploration of unknown actions with exploitation of learned best actions.

---

## ⚙️ Algorithm Flow

1. **Initialization**:
   - Discretize observation space into bins.
   - Initialize Q-table with zeros.

2. **Training Loop**:
   - For each episode:
     - Start at an initial state.
     - Choose action using ε-greedy policy.
     - Take action, observe reward and next state.
     - Update Q-table using Q-learning rule.
     - Decay ε over time for better exploitation.

3. **Evaluation**:
   - After training, evaluate the learned policy.
   - Plot episode rewards, Q-values, and visualize performance.

---

## 📊 Expected Outcomes

- Plots of cumulative rewards per episode.
- Q-value heatmaps showing learned policy.
- Understanding of:
  - How state discretization affects learning.
  - Impact of hyperparameters (learning rate, discount factor, ε-decay).
- Fully trained agent capable of solving MountainCar environment.

---

## 🌐 Useful References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- Gymnasium Documentation: https://gymnasium.farama.org/environments/classic_control/mountain_car/
- Grokking Deep Reinforcement Learning - Chapters on Q-learning and Value-based Methods.

---

## 💡 Notes

- Experiment with hyperparameters like learning rate (α), discount factor (γ), and exploration rate (ε).
- Track and visualize learning progression to better understand the dynamics of Q-learning.
