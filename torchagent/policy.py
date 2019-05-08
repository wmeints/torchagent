import numpy as np
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Policy:
    """ 
    Inherit from this class to implement a new policy.
    """

    def select_action(self, **kwargs):
        """
        Selects an action from the provided action space

        Parameters:
            kwargs (dict): The arguments for the policy

        Returns:
            The index of the selected action
        """
        raise NotImplementedError()


class GreedyPolicy(Policy):
    """
    The greedy policy doesn't use exploration at all, instead is chooses the best action based on the provided action-values.
    """

    def __init__(self, num_actions):
        """
        Initializes a new instance of the greedy policy.
        """
        super(GreedyPolicy, self).__init__()
        self.num_actions = num_actions

    def select_action(self, q_values):
        """
        Selects an action by choosing the index of the action with the highest q-value.
        """
        if len(q_values.size()) > 1:
            raise AssertionError("q_values has more than one dimension")

        action_tensor = q_values.max(len(q_values.size())-1)[1]
        action_tensor = action_tensor.view(1, 1)

        return action_tensor


class EpsilonGreedyPolicy(Policy):
    """
    The epsilon-greedy policy is a policy that uses a certain amount of randomness when selecting an action.
    This randomness is controlled by the epsilon setting. A high value means that we're going to explore more often,
    while a low value means that we're going to stick to what works and exploit the existing paths more.
    """

    def __init__(self, num_actions, epsilon=0.1):
        """
        Initializes the epsilon-greedy policy

        Parameters:
            epsilon (float): A number between 0 and 1 controlling the probability a random action is taken.
        """
        super(EpsilonGreedyPolicy, self).__init__()
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, q_values):
        """
        Selects an action from the provided action space

        Parameters:
            q_values: A tensor containing the predicted values for each action

        Returns:
            The index of the selected action
        """

        if np.random.uniform() < self.epsilon:
            selected_action = torch.tensor(
                [[random.randrange(self.num_actions)]],
                dtype=torch.long, device=device)
        else:
            selected_action = q_values.max(len(q_values.size())-1)[1]
            selected_action = selected_action.view(1, 1)

        return selected_action
