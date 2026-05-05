import pytest
from .cliff_walking import SARSACliffWalkingAgent, QLearningCliffWalkingAgent
from unittest.mock import MagicMock
from gymnasium.spaces import Discrete
import numpy as np


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.observation_space = Discrete(48)
    env.action_space = Discrete(4)
    return env


class TestSARSACliffWalkingAgent:
    @pytest.fixture
    def agent(self, mock_env):
        return SARSACliffWalkingAgent(mock_env, alpha=0.5, epsilon=0.1, min_epsilon=0.1, gamma=1.0, decay_rate=0.999)

    def test_act_returns_int(self, agent):
        assert isinstance(agent.act(0, eval=True), int)

    def test_update_nonterminal(self, agent):
        agent.q_table[:] = 0.0
        agent.q_table[1, 2] = 0.8
        agent.update(0, 0, -1.0, next_state=1, next_action=2)
        expected = 0.5 * (-1.0 + 1.0 * 0.8 - 0.0)
        assert abs(agent.q_table[0, 0] - expected) < 1e-9

    def test_update_terminal(self, agent):
        agent.q_table[:] = 0.0
        agent.update(0, 1, -100.0, next_state=None, next_action=None)
        expected = 0.5 * (-100.0 - 0.0)
        assert abs(agent.q_table[0, 1] - expected) < 1e-9


class TestQLearningCliffWalkingAgent:
    @pytest.fixture
    def agent(self, mock_env):
        return QLearningCliffWalkingAgent(mock_env, alpha=0.5, epsilon=0.1, min_epsilon=0.1, gamma=1.0, decay_rate=0.999)

    def test_update_uses_max(self, agent):
        agent.q_table[:] = 0.0
        agent.q_table[1] = [-5.0, -2.0, -1.0, -3.0]  # max is -1.0 at action 2
        agent.update(0, 0, -1.0, next_state=1)
        expected = 0.5 * (-1.0 + 1.0 * (-1.0) - 0.0)
        assert abs(agent.q_table[0, 0] - expected) < 1e-9

    def test_update_terminal(self, agent):
        agent.q_table[:] = 0.0
        agent.update(0, 1, -100.0, next_state=None)
        expected = 0.5 * (-100.0 - 0.0)
        assert abs(agent.q_table[0, 1] - expected) < 1e-9
