import random
from collections import deque, namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward',  'done'))


class SequentialMemory:
    """
    A sequential memory serves as a replay buffer with no priority rules.
    Items that are older are automatically removed when the replay memory
    reaches its maximum capacity.
    """

    def __init__(self, capacity):
        """
        Initializes a new instance of a sequential replay memory.
        This memory can store previous state changes in sequential order.
        """
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def append(self, state, action, next_state, reward, done):
        """
        Appends a transition to the replay memory

        Parameters:
            state (object): The current state
            action (object): The action that was performed
            next_state (object): The next state
            reward (object): The received reward
            done (boolean): Flag indicating the episode was completed
        """
        self.data.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size=1):
        """
        Samples transitions from the replay memory

        Parameters:
            batch_size (int): The number of samples to retrieve

        Returns:
            array: The array of sampled entries
        """
        if len(self.data) < batch_size:
            raise AssertionError('There are not enough samples available')

        return random.sample(self.data, batch_size)

    def __len__(self):
        """
        Calculates the size of the data stored in the replay memory

        Returns:
            int: The number of entries stored in the replay memory
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a single entry from the replay memory

        Parameters:
            index (int): The index of the item to retrieve
        
        Returns:
            Transition: The found transition
        """
        if index < 0 or index >= len(self.data):
            raise KeyError()

        return self.data[index]
