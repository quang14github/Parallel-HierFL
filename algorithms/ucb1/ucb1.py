import numpy as np

class UCB1:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)  # Number of times each arm has been pulled
        self.values = np.zeros(num_arms)  # Estimated values of each arm

    def select_arm(self):
        # Select arms with UCB1 criteria
        total_counts = np.sum(self.counts)
        if 0 in self.counts:
            # Choose arms that have not been explored yet
            return np.argmin(self.counts)
        else:
            # Calculate UCB1 values for each arm
            ucb_values = (self.values / self.counts) + np.sqrt(2 * np.log(total_counts) / self.counts)
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        # Update counts and values for the chosen arm
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += reward
        
    def log(self):
        print(self.counts)
        print(self.values)
