torchagent - A reinforcement learning library based on PyTorch
==============================================================
Welcome to the torchagent repository. This repository contains the sources
for the torchagent library.

.. contents::

What is it?
-----------
:code:`torchagent` is a library that implements various reinforcement learning algorithms for PyTorch.
You can use this library in combination with openAI Gym to implement reinforcement learning solutions.

Which algorithms are included?
------------------------------
Currently the following algorithms are implemented:

- Deep Q Learning 
- Double Q Learning

Installation
------------
You can install the library using the following command:

.. code::

    pip install torchagent

Usage
-----
The following code shows a basic agent that uses Deep Q Learning.

.. code:: python

    from torchagent.memory import SequentialMemory
    from torchagent.agents import DQNAgent

    import torch
    import torch.nn as nn
    import torch.optim as optim

    class PolicyNetwork(nn.Module):
        def __init__(self):
            self.linear = nn.Linear(210 * 160, 3)

        def forward(self, x):
            return self.linear(x)

    policy_network = PolicyNetwork()
    memory = SequentialMemory(20)
    agent = DQNAgent(2, policy_network, nn.MSELoss(), optim.Adam(policy_network.parameters()), memory)

    env = gym.make('Assault-v0')

    for _ in range(50):
        state = env.reset()

        for t in count():
            action = agent.act(state)
            next_state, reward, done, _ = env.step(agent.act(state))

            agent.record(state, action, next_state, reward, done)
            agent.train()

            state = next_state

            if done:
                break
