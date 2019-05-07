import pytest
from torchagent.memory import SequentialMemory


@pytest.fixture
def sequential_memory():
    return SequentialMemory(10)


def test_append_sequential_memory(sequential_memory):
    sequential_memory.append([0, 0, 0], 1, [0, 0, 1], 1, False)
    assert len(sequential_memory) == 1


def test_append_sequential_memory_rollover(sequential_memory):
    for i in range(1, 12):
        sequential_memory.append([0, 0, i], i, [0, 0, i+1], 1, False)

    assert len(sequential_memory) == 10
    assert sequential_memory[0].state == [0, 0, 2]


def test_sequential_memory_getitem(sequential_memory):
    sequential_memory.append([0, 0, 1], 1, [0, 0, 2], 1, False)

    assert sequential_memory[0].state == [0, 0, 1]
    assert sequential_memory[0].next_state == [0, 0, 2]


def test_batch_overflow_raises_error(sequential_memory):
    with pytest.raises(AssertionError):
        sequential_memory.sample(32)
