import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from abc import ABC, abstractmethod
from gymnasium.spaces.utils import flatdim


"""
Q-Learning assumes future behavior will be greedy (the max). So it estimates the value of the
cliff-edge path as if it will never randomly fall. It learns the theoretically optimal path
(shortest route), because its updates ignore the exploration noise.

SARSA accounts for the fact that its future self will sometimes explore randomly. The cliff-edge
path gets penalized because SARSA "knows" it will occasionally stumble off the edge. So it learns
to stay away — the longer but safer inland path.
"""


class FrozenLakeAgent(ABC):
    def __init__(self, env: FrozenLakeEnv):
        self.obs_space = env.observation_space
        self.act_space = env.action_space

    @abstractmethod
    def act(self, obs: int) -> int:
        pass


class RandomFrozenLakeAgent(FrozenLakeAgent):
    def act(self, obs: int) -> int:
        return self.act_space.sample()


class SARSAFrozenLakeAgent(FrozenLakeAgent):
    def __init__(self, env, alpha, epsilon, min_epsilon, gamma, decay_rate):
        super().__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.s_dim = flatdim(self.obs_space)
        self.a_dim = flatdim(self.act_space)

        self.q_table = np.ones([self.s_dim, self.a_dim])

    def act(self, obs: int, eval=True) -> int:
        if eval: return np.argmax(self.q_table[obs]).item()
        action = np.argmax(self.q_table[obs]).item() if np.random.random() > self.epsilon else np.random.randint(0, self.a_dim)
        self.epsilon = max(self.epsilon*self.decay_rate, self.min_epsilon)
        return action

    def update(self, state, action, reward, next_state, next_action):
        if next_state is None:
            self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
            )

    def __repr__(self) -> str:
        actions = ["L", "D", "R", "U"]
        grid_w = int(self.s_dim ** 0.5)
        header = "       " + "  ".join(f"{a:>7}" for a in actions)
        rows = [header]
        for s in range(self.s_dim):
            row, col = divmod(s, grid_w)
            label = f"({row},{col})"
            vals = "  ".join(f"{v:7.4f}" for v in self.q_table[s])
            rows.append(f"{label:>6} {vals}")

        rows.append("\nBest policy:")
        sep = "+" + "+".join(["---"] * grid_w) + "+"
        rows.append(sep)
        for r in range(grid_w):
            cells = []
            for c in range(grid_w):
                s = r * grid_w + c
                symbol = "·" if np.all(self.q_table[s] == 0) else actions[np.argmax(self.q_table[s])]
                cells.append(f" {symbol} ")
            rows.append("|" + "|".join(cells) + "|")
            rows.append(sep)

        return "\n".join(rows)


class QLearningFrozenLakeAgent(SARSAFrozenLakeAgent):
    def update(self, state, action, reward, next_state):
        if next_state is None:
            self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state].max() - self.q_table[state, action]
            )


def test_policy(agent: FrozenLakeAgent, name: str, env: Env, n_episodes: int = 10, record_video: bool = True, video_dir: str = "videos/tabular_q/frozen_lake/") -> float:
    if record_video:
        env = RecordVideo(env, video_folder=video_dir+name, episode_trigger=lambda _: True, disable_logger=True)

    total_rewards = 0
    with tqdm(range(n_episodes), desc=f"eval {name}") as pbar:
        for episode in pbar:
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards += episode_reward
            pbar.set_postfix(win_rate=f"{total_rewards / (episode + 1):.1%}")

    env.close()
    avg_rew = total_rewards / n_episodes
    print(f"Win rate: {total_rewards / n_episodes:.1%}")
    return avg_rew

def train_sarsa(agent: SARSAFrozenLakeAgent, env: Env, n_episode: int = 1000, verbose: bool = True) -> list[float]:
    rewards = []
    with tqdm(range(n_episode), desc="train SARSA", disable=not verbose) as pbar:
        for ep in pbar:
            obs, _ = env.reset()
            action = agent.act(obs, eval=False)
            done = False
            ep_reward = 0.0

            while not done:
                prev_obs, prev_action = obs, action
                obs, rew, ter, tru, _ = env.step(action)
                done = ter or tru
                ep_reward += rew
                next_action = agent.act(obs, eval=False) if not done else None
                agent.update(prev_obs, prev_action, rew, None if done else obs, next_action)
                action = next_action

            rewards.append(ep_reward)
            pbar.set_postfix(epsilon=f"{agent.epsilon:.4f}")
    return rewards


def train_qlearning(agent: QLearningFrozenLakeAgent, env: Env, n_episode: int = 1000, verbose: bool = True) -> list[float]:
    rewards = []
    with tqdm(range(n_episode), desc="train Q-learning", disable=not verbose) as pbar:
        for ep in pbar:
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                prev_obs = obs
                action = agent.act(obs, eval=False)
                obs, rew, ter, tru, _ = env.step(action)
                done = ter or tru
                ep_reward += rew
                agent.update(prev_obs, action, rew, None if done else obs)

            rewards.append(ep_reward)
            pbar.set_postfix(epsilon=f"{agent.epsilon:.4f}")
    return rewards


def plot_training_curves(
    histories: dict[str, list[float]],
    window: int = 5000,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rewards in histories.items():
        arr = np.array(rewards, dtype=float)
        kernel = np.ones(window) / window
        smoothed = np.convolve(arr, kernel, mode="valid")
        ax.plot(np.arange(len(smoothed)) + window - 1, smoothed, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Avg reward (window={window})")
    ax.set_title("Training progress")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    video_dir = "videos/tabular_q/frozen_lake/"
    record_video = True
    test_n_episode = 10
    n_train = 500000

    # Env param
    slippery = True
    success_rate = 3.0/5.0

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, success_rate=success_rate, reward_schedule=(1, 0, 0), render_mode="rgb_array" if record_video else None)

    # random agent
    random_agent = RandomFrozenLakeAgent(env)
    test_policy(agent=random_agent, name="random_agent", env=env, n_episodes=test_n_episode, record_video=record_video, video_dir=video_dir)

    # SARSA agent
    sarsa_agent = SARSAFrozenLakeAgent(env, alpha=0.2, epsilon=1.0, min_epsilon=0.1, gamma=0.9, decay_rate=0.999999)
    sarsa_rewards = train_sarsa(sarsa_agent, env, n_episode=n_train)
    test_policy(agent=sarsa_agent, name="sarsa_agent", env=env, n_episodes=test_n_episode, record_video=record_video, video_dir=video_dir)

    # Q-learning agent
    ql_agent = QLearningFrozenLakeAgent(env, alpha=0.2, epsilon=1.0, min_epsilon=0.1, gamma=0.9, decay_rate=0.999999)
    ql_rewards = train_qlearning(ql_agent, env, n_episode=n_train)
    test_policy(agent=ql_agent, name="ql_agent", env=env, n_episodes=test_n_episode, record_video=record_video, video_dir=video_dir)

    plot_training_curves(
        {"SARSA": sarsa_rewards, "Q-learning": ql_rewards},
        window=5000,
        save_path="videos/tabular_q/frozen_lake/training_curves.png",
    )
