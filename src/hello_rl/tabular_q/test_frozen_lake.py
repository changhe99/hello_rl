import pytest
from .frozen_lake import SARSAFrozenLakeAgent, QLearningFrozenLakeAgent
from unittest.mock import MagicMock, patch
from gymnasium.spaces import Discrete
import numpy as np

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.observation_space = Discrete(16)
    env.action_space = Discrete(4)
    return env

class TestSARSAFrozenLakeAgent:
    @pytest.fixture
    def agent(self, mock_env):
        return SARSAFrozenLakeAgent(mock_env, 0.1, 0.1, 0.1, 0.9, 0.99)

    def test_ql_act_dtype(self, agent):
        action = agent.act(0, eval=True)
        assert isinstance(action, int)

    def test_ql_random_dtype(self, agent):
        with patch("hello_rl.tabular_q.frozen_lake.np.random.randint", side_effect=lambda *args: 3) as mock_randint, \
            patch("hello_rl.tabular_q.frozen_lake.np.random.random", side_effect=lambda: 0.05):
            action = agent.act(0, eval=False)
            mock_randint.assert_called_once_with(0, 4)
            assert action == 3

    def test_sarsa_update_nonterminal(self, agent):
        # SARSA uses Q(s', a'), not max Q(s', *)
        agent.q_table[:] = 0.0
        agent.q_table[1, 2] = 0.5
        agent.update(0, 0, 1.0, next_state=1, next_action=2)
        expected = 0.1 * (1.0 + 0.9 * 0.5 - 0.0)
        assert abs(agent.q_table[0, 0] - expected) < 1e-9

    def test_sarsa_update_terminal(self, agent):
        agent.q_table[:] = 0.0
        agent.update(0, 1, 1.0, next_state=None, next_action=None)
        expected = 0.1 * (1.0 - 0.0)
        assert abs(agent.q_table[0, 1] - expected) < 1e-9


class TestQLearningFrozenLakeAgent:
    @pytest.fixture
    def agent(self, mock_env):
        return QLearningFrozenLakeAgent(mock_env, 0.1, 0.1, 0.1, 0.9, 0.99)

    def test_ql_update_uses_max(self, agent):
        # Q-learning uses max Q(s', *), not Q(s', a')
        agent.q_table[:] = 0.0
        agent.q_table[1] = [0.1, 0.2, 0.5, 0.3]  # max is 0.5 at action 2
        agent.update(0, 0, 1.0, next_state=1)
        expected = 0.1 * (1.0 + 0.9 * 0.5 - 0.0)
        assert abs(agent.q_table[0, 0] - expected) < 1e-9

    def test_ql_update_terminal(self, agent):
        agent.q_table[:] = 0.0
        agent.update(0, 1, 1.0, next_state=None)
        expected = 0.1 * (1.0 - 0.0)
        assert abs(agent.q_table[0, 1] - expected) < 1e-9
