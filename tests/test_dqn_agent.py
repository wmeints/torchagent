import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from torchagent.agents import DQNAgent
from torchagent.memory import SequentialMemory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()

        self.linear1 = nn.Linear(10, 10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 2)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.linear2(self.relu1(self.linear1(x))))

@pytest.fixture()
def model():
    return SampleModel().to(device)

@pytest.fixture()
def agent(model):
    return DQNAgent(2, model, nn.MSELoss(), optim.Adam(model.parameters()), SequentialMemory(10))

@pytest.fixture()
def double_q_agent(model):
    return DQNAgent(2, model, nn.MSELoss(), optim.Adam(model.parameters()), SequentialMemory(10), enable_dqn=True)

def test_agent_predict(agent):
    state = torch.from_numpy(np.random.random((1,10)).astype(np.float32)).to(device)
    result = agent.act(state)

    assert result > -1

def test_agent_train_double_q(double_agent):
    for _ in range(0, 11):
        state = torch.from_numpy(np.random.random((1,10)).astype(np.float32)).to(device)
        next_state = torch.from_numpy(np.random.random((1,10)).astype(np.float32)).to(device)
        action = double_agent.act(state)

        double_agent.record(state, action, next_state, 1, False)

    double_agent.train()

def test_agent_train(agent):
    for _ in range(0, 11):
        state = torch.from_numpy(np.random.random((1,10)).astype(np.float32)).to(device)
        next_state = torch.from_numpy(np.random.random((1,10)).astype(np.float32)).to(device)
        action = agent.act(state)

        agent.record(state, action, next_state, 1, False)

    agent.train()

