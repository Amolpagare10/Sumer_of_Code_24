# 📚 Week 3 - Multi-Armed Bandits and Reinforcement Learning

## 🚀 Overview

In **Week 3**, we dive deep into the **Multi-Armed Bandit problem**, one of the foundational problems in Reinforcement Learning. The focus is on understanding **exploration vs. exploitation trade-offs**, implementing various policies, and analyzing the performance of these algorithms. This week lays the groundwork for working with unknown environments where optimal actions need to be discovered over time.

---

## 🎯 Objectives

- ✅ Understand the **Multi-Armed Bandit** problem setup and objectives.
- ✅ Implement and analyze different policies to solve Bandit problems.
- ✅ Study **regret minimization** and evaluate agent performance.
- ✅ Gain hands-on experience with algorithm design and performance visualization.

---

## 📦 File Structure
```txt
Week_3/
├── bandits.py    # Bandit class with Bernoulli and Gaussian reward distributions, regret tracking.
├── agents.py     # Agent classes implementing different strategies (Epsilon-Greedy, UCB, Thompson Sampling, etc.).
└── results.py    # Scripts for training agents, plotting learning curves, and analyzing performance.
```
---

## 🧠 Concepts Covered

- **Multi-Armed Bandits (MAB)**: 
  - A scenario where an agent chooses between multiple actions (arms) to maximize cumulative rewards.
  - Each arm provides stochastic rewards, and the agent must balance **exploration** (trying new arms) vs. **exploitation** (choosing known good arms).

- **Reward Distributions**:
  - **Bernoulli Bandits**: Rewards are binary (0 or 1), returned with a fixed probability.
  - **Gaussian Bandits**: Rewards are drawn from a Gaussian distribution with a fixed mean and unit variance.

- **Policies Implemented**:
  - **Epsilon-Greedy**: Explores random arms with probability ε, otherwise chooses the best-known arm.
  - **Upper Confidence Bound (UCB)**: Selects arms based on optimistic estimates of their rewards.
  - **Thompson Sampling**: Uses Bayesian inference (Beta distribution as a conjugate prior) to balance exploration and exploitation.

- **Regret Tracking**:
  - Measures the difference between the reward accumulated by the optimal arm and the agent's chosen arms.
  - Formula:
    \[
    \text{Regret}(t) = k \times \text{Optimal Reward} - \sum_{i=1}^{t} R_i
    \]

---

## 🛠️ How to Use

1. **Define the Bandit Problem** in `bandits.py` by selecting reward distributions and number of arms.
2. **Implement Agent Policies** in `agents.py` to interact with the bandit environment.
3. **Train and Evaluate Agents** using `results.py` to plot cumulative rewards, regrets, and performance comparisons.

---

## 📊 Expected Outcomes

- Comparative plots of cumulative rewards and regrets for different strategies.
- Analysis of exploration vs. exploitation performance.
- Understanding of how different algorithms handle the trade-off between trying new actions and leveraging known good actions.

