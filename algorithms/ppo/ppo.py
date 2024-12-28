import torch.optim as optim
from .network import *
from .replay_buffer import *


class Agent:
    def __init__(
        self,
        state_dims,
        hidden_dims,
        action_dims,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
        max_memory_size=10000,
    ):
        self.policy_network = PolicyNetwork(state_dims, hidden_dims, action_dims)
        self.value_network = ValueNetwork(state_dims, hidden_dims)
        self.epochs = epochs
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.memory = Memory(max_memory_size)
        self.policy_optim = optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )
        self.value_optim = optim.Adam(
            self.value_network.parameters(), lr=self.learning_rate
        )

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        return (
            action.item(),
            action_probs.log_prob(action).detach(),
            action_probs.entropy(),
        )

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        return self.value_network(state)

    def learn(self, batch_size):
        for _ in range(self.epochs):
            states, actions, log_probs, rewards, next_states = self.memory.sample(
                batch_size
            )
            states = torch.tensor(np.array(states), dtype=torch.float)
            actions = torch.tensor(np.array(actions), dtype=torch.float)
            log_probs = torch.tensor(log_probs, dtype=torch.float).flatten()
            rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float)

            # calculate next value, advantage
            values = self.value_network(states).squeeze()
            next_values = self.value_network(next_states).squeeze()
            returns = rewards + self.gamma * next_values
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            new_probs = self.policy_network(states)
            new_action_probs = torch.distributions.Categorical(new_probs)
            new_log_probs = new_action_probs.log_prob(actions)

            ratio = (new_log_probs - log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
            value_loss = F.mse_loss(values, returns)

            # Calculate gradients and perform backward propagation for actor network
            self.policy_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
