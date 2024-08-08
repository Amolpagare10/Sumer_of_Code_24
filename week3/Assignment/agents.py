from bandits import Bandit
import numpy as np

class Agent:
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.banditN = bandit.getN()
        self.rewards = 0
        self.numiters = 0

    def action(self) -> int:
        '''This function returns which action is to be taken. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    def update(self, choice: int, reward: int) -> None:
        '''This function updates all member variables you may require. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    # don't edit this function
    def act(self) -> int:
        choice = self.action()
        reward = self.bandit.choose(choice)
        self.rewards += reward
        self.numiters += 1
        self.update(choice, reward)
        return reward

class GreedyAgent(Agent):
    def __init__(self, bandit: Bandit, initialQ: float) -> None:
        super().__init__(bandit)
        self.Q = np.full(self.banditN, initialQ)  # Q-value estimates for each action
        self.action_counts = np.zeros(self.banditN)  # Count of each action taken
    
    def action(self) -> int:
        # Choose action with the highest Q-value
        return np.argmax(self.Q)

    def update(self, choice: int, reward: int) -> None:
        self.action_counts[choice] += 1
        self.Q[choice] += (reward - self.Q[choice]) / self.action_counts[choice]

class epsGreedyAgent(Agent):
    def __init__(self, bandit: Bandit, epsilon: float) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        self.Q = np.zeros(self.banditN)
        self.action_counts = np.zeros(self.banditN)
    
    def action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.banditN)  # Explore: random action
        else:
            return np.argmax(self.Q)  # Exploit: best current action

    def update(self, choice: int, reward: int) -> None:
        self.action_counts[choice] += 1
        self.Q[choice] += (reward - self.Q[choice]) / self.action_counts[choice]

class UpperConfidenceBoundAgent(Agent):
    def __init__(self, bandit: Bandit, c: float) -> None:
        super().__init__(bandit)
        self.c = c
        self.Q = np.zeros(self.banditN)
        self.action_counts = np.zeros(self.banditN)
        self.total_counts = 0

    def action(self) -> int:
        if 0 in self.action_counts:
            return np.argmin(self.action_counts)  # Choose action with zero counts
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.total_counts + 1) / self.action_counts)
        return np.argmax(ucb_values)

    def update(self, choice: int, reward: int) -> None:
        self.total_counts += 1
        self.action_counts[choice] += 1
        self.Q[choice] += (reward - self.Q[choice]) / self.action_counts[choice]

class GradientBanditAgent(Agent):
    def __init__(self, bandit: Bandit, alpha: float) -> None:
        super().__init__(bandit)
        self.alpha = alpha
        self.preferences = np.zeros(self.banditN)
        self.action_probs = np.ones(self.banditN) / self.banditN
        self.avg_reward = 0

    def action(self) -> int:
        self.action_probs = np.exp(self.preferences) / np.sum(np.exp(self.preferences))
        return np.random.choice(self.banditN, p=self.action_probs)

    def update(self, choice: int, reward: int) -> None:
        self.avg_reward += (reward - self.avg_reward) / (self.numiters + 1)
        baseline = reward - self.avg_reward
        for a in range(self.banditN):
            if a == choice:
                self.preferences[a] += self.alpha * baseline * (1 - self.action_probs[a])
            else:
                self.preferences[a] -= self.alpha * baseline * self.action_probs[a]

class ThompsonSamplerAgent(Agent):
    def __init__(self, bandit: Bandit) -> None:
        super().__init__(bandit)
        if bandit.type == "Bernoulli":
            self.alpha = np.ones(bandit.getN())
            self.beta = np.ones(bandit.getN())
        else:
            raise ValueError("Thompson Sampling only supports Bernoulli bandits.")

    def action(self) -> int:
        samples = [np.random.beta(self.alpha[a], self.beta[a]) for a in range(self.banditN)]
        return np.argmax(samples)

    def update(self, choice: int, reward: int) -> None:
        self.alpha[choice] += reward
        self.beta[choice] += 1 - reward
