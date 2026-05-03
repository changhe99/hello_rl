import pytest
from .frozen_lake import QLFrozenLakeAgent
from unittest.mock import MagicMock, patch
from gymnasium.spaces import Discrete
import numpy as np

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.observation_space = Discrete(16)
    env.action_space = Discrete(4)
    return env

class TestQLFrozenLakeAgent:
    @pytest.fixture
    def ql(self, mock_env):
        return QLFrozenLakeAgent(mock_env, 0.1, 0.1, 0.1, 0.9, 0.99)

    def test_ql_act_dtype(self, ql):
        obs = 0
        action = ql.act(obs, eval=True)
        assert isinstance(action, int)

    def test_ql_random_dtype(self, ql):
        obs = 0
        with patch("hello_rl.tabular_q.frozen_lake.np.random.randint", side_effect=lambda *args: 3) as mock_randint, \
            patch("hello_rl.tabular_q.frozen_lake.np.random.random", side_effect=lambda: 0.05):
            action = ql.act(obs, eval=False)
            mock_randint.assert_called_once_with(0, 4)
            assert action == 3

    def test_ql_update(self, ql):
        ql.update(0, 0, -10, 1)
        ql.update(0, 1, 10, None)
