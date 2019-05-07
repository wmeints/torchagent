import torch
import numpy as np
import pytest
from torchagent.policy import EpsilonGreedyPolicy

@pytest.fixture
def policy():
    return EpsilonGreedyPolicy()

def test_select_action(policy):
    output = policy.select_action(torch.from_numpy(np.array([0,1,2,5])))
    assert output is not None