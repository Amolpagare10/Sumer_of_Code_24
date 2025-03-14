# 📊 Reinforcement Learning & Applied ML Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Reinforcement%20Learning-orange)

</div>

---

## 🚀 About the Project

This repository contains a **4-week learning series** of applied **Machine Learning and Reinforcement Learning (RL)** algorithms. During this period, I explored concepts ranging from **probabilistic methods, bandit problems, Markov Decision Processes, Monte Carlo methods, Temporal Difference learning to Q-learning**, implementing all solutions from scratch.  

The entire work focuses on **building strong foundational understanding** and **hands-on implementation** of key RL algorithms and ML techniques, including **environment interaction using Gymnasium**, dynamic programming, and value function approximation.  

---

## 🗓️ Weekly Breakdown

### 📅 Week 1: Probability and Linear Algebra for ML
- **Inverse Transform Sampling**:  
  A technique to generate random numbers that follow a given probability distribution when direct sampling is not feasible.  
- **Principal Component Analysis (PCA)**:  
  Dimensionality reduction technique to project data onto orthogonal components capturing maximum variance.  
- **Curve Fitting**:  
  Applied curve fitting methods to model data using functional approximations, optimizing fit parameters.  

---

### 📅 Week 2: Introduction to Reinforcement Learning (RL)
- **N-Armed Bandits**:  
  Studied the classic exploration-exploitation dilemma in RL, implemented algorithms to find arms with maximum expected reward.  
- **Markov Decision Processes (MDP)**:  
  Formalized RL problems using MDPs, understanding states, actions, rewards, and transition probabilities.  
- **Dynamic Programming (DP)**:  
  Explored DP methods for solving MDPs when the model is known. Focused on **policy evaluation**, **policy improvement**, and **value iteration** for finding optimal policies.

---

### 📅 Week 3: Model-Free Prediction & Control
- **Monte Carlo Methods**:  
  Learned sampling-based methods to estimate value functions and solve control problems without knowing environment dynamics.  
- **Temporal Difference (TD) Learning**:  
  Combined ideas of Monte Carlo and DP to learn directly from raw experience without full model knowledge.  
- **Reading Material**:  
  - Chapters 5 & 6 of *Grokking Deep Reinforcement Learning*.  
  - Sutton & Barto’s *Reinforcement Learning: An Introduction* (Chapters 5, 6, and optionally Chapter 7 on Eligibility Traces).

#### ✅ Assignment: Multi-Armed Bandit Problem
- Implemented various policies for **Multi-Armed Bandits** in a modular structure:  
  - **bandits.py**: Bandit class supporting Bernoulli and Gaussian arms, with regret tracking.
  - **agents.py**: Agent classes implementing different strategies (e.g., epsilon-greedy, UCB, Thompson Sampling).
  - **results.py**: Training, visualization, and performance plots of the agents.  

---

### 📅 Week 4: Q-Learning and Solving Gymnasium Environments
- **Q-Learning**:  
  Implemented Q-Learning to solve the **Mountain Car** environment (continuous observation space) using **state discretization**.
- **Gymnasium Environment**:  
  Leveraged Gymnasium to simulate environments for RL problems.  
- **Mountain Car Problem**:  
  - Task: Drive a car up a steep hill with limited power.
  - Strategy:  
    1. **Discretize** continuous state space.  
    2. Implement **epsilon-greedy policy** for action selection.  
    3. Apply **Q-learning update rule**:  
    \[
    Q(s, a) \leftarrow Q(s, a) + \alpha (r + \gamma \max_{a'} Q(s', a') - Q(s, a))
    \]
  - Experimented with different hyperparameters and evaluated agent learning performance.

#### ✅ Assignment:
- **State-Action Value Approximation** using Q-learning.  
- Analysis of agent learning performance via plotted metrics like average reward per episode, exploration vs. exploitation, etc.  
- Saving Q-tables periodically for analysis.

---

## 📂 Project Structure
```txt
├── Week_1
│   ├── inverse_transform_sampling.py      # Implementation of Inverse Transform Sampling
│   ├── pca.py                             # Principal Component Analysis (PCA) implementation
│   └── curve_fitting.py                   # Curve fitting techniques
│
├── Week_2
│   ├── n_armed_bandit.py                  # N-Armed Bandit problem and solution algorithms
│   └── mdp_dynamic_programming.py         # Markov Decision Process and Dynamic Programming methods
│
├── Week_3
│   ├── bandits.py                        # Bandit environment implementation (Bernoulli, Gaussian)
│   ├── agents.py                        # Agents for different bandit algorithms (Epsilon-Greedy, UCB, Thompson Sampling, etc.)
│   └── results.py                        # Training and plotting results of different agents
│
└── Week_4
    ├── q_learning_mountain_car.py         # Q-Learning for Mountain Car environment (Gymnasium)
    ├── utils.py                          # Helper functions (state discretization, epsilon scheduling, etc.)
    └── analysis_plots.py                  # Performance visualization and analysis of learning agent
```

---

## 📊 Key Results & Visualizations

- ✅ **Exploration vs. Exploitation** trade-offs in Multi-Armed Bandits.
- ✅ **Regret minimization** performance comparison between policies.
- ✅ **Q-learning convergence plots** for Mountain Car.
- ✅ Impact of **hyperparameter tuning** on learning efficiency.

---

## 📚 References

- **Grokking Deep Reinforcement Learning** by Miguel Morales.
- **Reinforcement Learning: An Introduction** by Richard S. Sutton and Andrew G. Barto (2nd Edition).
- **Gymnasium Documentation**: [https://gymnasium.farama.org](https://gymnasium.farama.org).
- Numpy and Matplotlib libraries for numerical computations and plotting.




