import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    # This Class is for a n-armed bandit algorithm.

    def __init__(self, n, epsilon, steps=1000):
        """
        In the class, n is the numers of arms, epsilon as e is the exploration
        probability, steps is the time step, q_star is the true reward values
        for each arm, Q is the estimated values, action_counts is the number
        of times each arm is selected, rewards is the rewards obtained over
        time, opt_action_counts is the count of optimal action selections,
        and the optimal_action is the best possible action.
        """

        self.n = n
        self.e = epsilon
        self.steps = steps
        self.q_star = np.random.normal(0, 1, n)
        self.Q = np.zeros(n)
        self.action_counts = np.zeros(n)
        self.rewards = []
        self.optimal_action = np.argmax(self.q_star)
        self.opt_action_counts = []

    def action(self):
        # This function deals with actions, exploration and exploitation

        if np.random.rand() < self.e:
            return np.random.randint(self.n)
        else:
            return np.argmax(self.Q)

    def actionValued(self):
        # This function returns reward from normal distribution

        action = self.action()
        reward = np.random.normal(self.q_star[action], 1)
        self.rewards.append(reward)
        self.action_counts[action] += 1
        self.Q[action] += (reward - self.Q[action])/self.action_counts[action]
        self.opt_action_counts.append(int(action == self.optimal_action))

    def update(self):
        # This function updates rewards and optimal action counts

        for _ in range(self.steps):
            self.actionValued()
        return np.array(self.rewards), np.array(self.opt_action_counts)


def runArmedBandit(n_values=[5, 10, 20], epsilon_lst=[0, 0.1, 0.01], steps=1000, runs=2000):
    """
    This function dispalys the Multi Armed Bandit Algorithm for different
    values of n, epsilons, steps and runs number of plays.
    Left column shows Average Reward and Right column displays Optimal Action.
    """

    plt.rcParams["font.family"] = "Cambria"
    cmfont = {'fontname': 'Cambria'}
    color_map = {0: 'green', 0.1: 'blue', 0.01: 'm'}
    fig, axes = plt.subplots(len(n_values), 2, figsize=(14, 21))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.5)

    for i, n in enumerate(n_values):
        all_avg_rewards = {}
        all_opt_action_perc = {}

        for e in epsilon_lst:
            all_rewards = np.zeros(steps)
            all_optimal = np.zeros(steps)

            for _ in range(runs):
                bandit = MultiArmedBandit(n, e, steps)
                rewards, optimal_actions = bandit.update()
                all_rewards += rewards
                all_optimal += optimal_actions

            avg_rewards = all_rewards / runs
            opt_action_percent = all_optimal / runs * 100

            all_avg_rewards[e] = avg_rewards
            all_opt_action_perc[e] = opt_action_percent

            axes[i, 0].plot(avg_rewards, label=f'ε={e}', color=color_map[e], linewidth=1)
            axes[i, 1].plot(opt_action_percent, label=f'ε={e}', color=color_map[e], linewidth=1.2)

        axes[i, 0].set_title(f'n={n}', **cmfont)
        axes[i, 0].set_xlabel('Steps', **cmfont)
        axes[i, 0].set_ylabel('Average Reward', **cmfont)
        axes[i, 0].legend()

        axes[i, 1].set_title(f'n={n}', **cmfont)
        axes[i, 1].set_xlabel('Steps', **cmfont)
        axes[i, 1].set_ylabel('% Optimal Action', **cmfont)
        axes[i, 1].legend()

    plt.suptitle('Multi-Armed Bandit Algorithm Performance')
    plt.show()


runArmedBandit()
