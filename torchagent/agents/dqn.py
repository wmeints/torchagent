import copy
import torch

from torchagent.policy import EpsilonGreedyPolicy, GreedyPolicy
from torchagent.memory import Transition, SequentialMemory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQNAgent:
    """
    An implementation of an agent that uses Deep Q-Learning.
    """

    def __init__(self, num_actions, model, loss, optimizer,
                 memory, policy=None, test_policy=None,
                 training=True, enable_dqn=False, batch_size=32,
                 gamma=0.9, tau=1e-3):
        """
        Initializes a new instance of a Deep Q-Learning agent.

        Parameters:
            num_actions (int): The number of actions that is supported
            model (object): The neural network that is used to calculate the q-values for the agent
            loss (object): The loss function to use for optimizing the agent
            optimizer (object): The optimizer to use for optimizing the agent
            memory (object): The experience buffer to use for the agent
            policy (object): The policy network to use for the agent
            test_policy (object): The target policy network to use for the agent
            training (boolean): Flag indicating the agent is training
            enable_dqn (boolean): Flag enabling double-q value learning
            batch_size (int): The number of samples to use for each cycle of training
            gamma (float): The discount factor for rewards the agent receives
            tau (float): Factor controlling the speed at which the target model is updated
        """

        self.policy = policy if policy is not None else EpsilonGreedyPolicy(num_actions)
        self.test_policy = test_policy if test_policy is not None else GreedyPolicy(num_actions)
        self.memory = memory
        self.training = training
        self.batch_size = batch_size
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.enable_dqn = enable_dqn
        self.num_actions = num_actions
        self.loss = loss
        self.gamma = gamma
        self.optimizer = optimizer
        self.tau = tau
        self.step = 0

    def record(self, state, action, next_state, reward, done):
        """
        Records experience for the agent

        Parameters:
            state (object): The current state
            action (object): The action that was performed
            next_state (object): The next state
            reward (object): The received reward
            done (boolean): Flag indicating the episode was completed
        """
        self.memory.append(state, action, next_state, reward, done)

    def act(self, observation):
        """
        Allows the agent to perform a step based on the provided observation of the environment.

        Parameters:
            observation (object): The observation for the agent to base its decision on.

        Returns:
            int: The index of the selected action
        """

        # First, calculate the q-values for the action space that the agent supports
        # Next, use the training or test policy to predict the next action to take
        with torch.no_grad():
            action_tensor = self.model(observation)

        if self.training:
            selected_action = self.policy.select_action(action_tensor, step=self.step)
        else:
            selected_action = self.test_policy.select_action(action_tensor, step=self.step)

        return selected_action

    def train(self):
        """
        Performs a single training pass using the agent's memory as input.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        self.step += 1

        self.train_model_(batch)
        self.update_target_model_()

    def train_model_(self, batch):
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminated_batch = torch.tensor(
            [1. if s == False else 0. for s in batch.done], device=device, dtype=torch.float)

        # We're trying to minimize a loss that is defined as follows:
        # loss = sum(sqr(reward + gamma * target_Q(next_state) - Q(state)))
        #
        # Please note that you can use the huber loss also, but it uses the same inputs:
        # - Q(next_state, a) --> Q-values for the target state
        # - Q(state, a) --> Q-values for the current state
        #
        # We're using a second network to calculate the Q-values for the target state.
        # this stabalizes the training process so it becomes more predictable.
        
        if not self.enable_dqn:
            target_q_values = self.target_model(next_state_batch)
            target_q_values = target_q_values.gather(1, action_batch)
            q_values = self.model(state_batch)
            q_values = q_values.gather(1, action_batch)
        else:
            # In double-q learning we use the policy network to predict the actions instead of using the actually performed actions.
            # We then use these predicted actions to calculate the expected value of these actions.
            q_values = self.target_model(state_batch)
            estimated_actions = q_values.max(1)[1]

            target_q_values = self.target_model(next_state_batch)
            target_q_values = target_q_values.gather(1, estimated_actions)

        # Calculate the discounted award for the actions taken by the agent.
        # Then reset the reward for the actions that caused the episode to end.
        discounted_reward = self.gamma * target_q_values
        discounted_reward = terminated_batch * discounted_reward

        # Calculate the expected reward using the discounted reward and the actual reward.
        targets = discounted_reward + reward_batch
        targets = targets.detach()

        # Now calculate the loss for the policy that we're currently using.
        loss_value = self.loss(q_values, targets.unsqueeze(1))

        # Finally, optimize the policy using the choosen optimizer.
        self.optimizer.zero_grad()
        loss_value.backward()

        for param in self.model.parameters():
            param.grad.data.clamp(-1, 1)

        self.optimizer.step()

    def update_target_model_(self):
        if self.tau < 1.:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) +
                    param.data + self.tau
                )
        else:
            if self.step % self.tau == 0:
                self.target_model.load_state_dict(self.model.state_dict())
