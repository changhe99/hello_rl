import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.core import Env

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from abc import ABC, abstractmethod
from gymnasium.spaces.utils import flatdim

# CliffWalking-v0: 4×12 grid, dense rewards (-1/step, -100/cliff)
# Classic SARSA vs Q-learning demo: SARSA learns safe path, Q-learning learns cliff-edge optimal path


class CliffWalkingAgent(ABC):
    def __init__(self, env):
        self.obs_space = env.observation_space
        self.act_space = env.action_space

    @abstractmethod
    def act(self, obs: int) -> int:
        pass


class RandomCliffWalkingAgent(CliffWalkingAgent):
    def act(self, obs: int) -> int:
        return self.act_space.sample()


class SARSACliffWalkingAgent(CliffWalkingAgent):
    N_ROWS, N_COLS = 4, 12
    # UP=0, RIGHT=1, DOWN=2, LEFT=3 (gymnasium convention)
    _ACTION_SYMBOLS = ["U", "R", "D", "L"]

    def __init__(self, env, alpha, epsilon, min_epsilon, gamma, decay_rate):
        super().__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.s_dim = flatdim(self.obs_space)
        self.a_dim = flatdim(self.act_space)

        self.q_table = np.zeros([self.s_dim, self.a_dim])

    def act(self, obs: int, eval=True) -> int:
        if eval:
            return np.argmax(self.q_table[obs]).item()
        action = np.argmax(self.q_table[obs]).item() if np.random.random() > self.epsilon else np.random.randint(0, self.a_dim)
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)
        return action

    def update(self, state, action, reward, next_state, next_action):
        if next_state is None:
            self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
            )

    def __repr__(self) -> str:
        sep = "+" + "+".join(["---"] * self.N_COLS) + "+"
        rows = [sep]
        for r in range(self.N_ROWS):
            cells = []
            for c in range(self.N_COLS):
                s = r * self.N_COLS + c
                # bottom row: S=start, G=goal, cliff markers
                if r == self.N_ROWS - 1 and 0 < c < self.N_COLS - 1:
                    cells.append(" C ")
                elif r == self.N_ROWS - 1 and c == 0:
                    cells.append(" S ")
                elif r == self.N_ROWS - 1 and c == self.N_COLS - 1:
                    cells.append(" G ")
                else:
                    cells.append(f" {self._ACTION_SYMBOLS[np.argmax(self.q_table[s])]} ")
            rows.append("|" + "|".join(cells) + "|")
            rows.append(sep)
        return "\n".join(rows)


class QLearningCliffWalkingAgent(SARSACliffWalkingAgent):
    def update(self, state, action, reward, next_state):
        if next_state is None:
            self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state].max() - self.q_table[state, action]
            )


def test_policy(
    agent: CliffWalkingAgent,
    name: str,
    env_id: str,
    n_episodes: int = 10,
    record_video: bool = True,
    video_dir: str = "videos/tabular_q/cliff_walking/",
    max_episode_steps: int = 200,
) -> float:
    eval_env = gym.make(env_id, render_mode="rgb_array" if record_video else None, max_episode_steps=max_episode_steps)
    if record_video:
        eval_env = RecordVideo(eval_env, video_folder=video_dir + name, episode_trigger=lambda _: True, disable_logger=True)

    total_rewards = 0.0
    with tqdm(range(n_episodes), desc=f"eval {name}") as pbar:
        for episode in pbar:
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards += episode_reward
            pbar.set_postfix(avg_rew=f"{total_rewards / (episode + 1):.1f}")

    eval_env.close()
    avg_rew = total_rewards / n_episodes
    print(f"Avg reward: {avg_rew:.1f}")
    return avg_rew


def train_sarsa(agent: SARSACliffWalkingAgent, env: Env, n_episode: int = 500) -> list[float]:
    rewards = []
    with tqdm(range(n_episode), desc="train SARSA") as pbar:
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


def train_qlearning(agent: QLearningCliffWalkingAgent, env: Env, n_episode: int = 500) -> list[float]:
    rewards = []
    with tqdm(range(n_episode), desc="train Q-learning") as pbar:
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
    window: int = 10,
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
    ax.set_title("Training progress — CliffWalking")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    video_dir = "videos/tabular_q/cliff_walking/"
    record_video = True
    test_n_episode = 10
    n_train = 1000

    # decay epsilon from 1.0 → 0.1 over ~500 episodes × ~20 steps/ep = ~10000 act calls
    decay_rate = 0.1 ** (1 / 10000)

    env_id = "CliffWalking-v1"
    env = gym.make(env_id, max_episode_steps=200)

    # SARSA — on-policy: learns safe path one row above the cliff
    sarsa_agent = SARSACliffWalkingAgent(env, alpha=0.5, epsilon=1.0, min_epsilon=0.1, gamma=1.0, decay_rate=decay_rate)
    sarsa_rewards = train_sarsa(sarsa_agent, env, n_episode=n_train)
    print(sarsa_agent)
    test_policy(agent=sarsa_agent, name="sarsa_agent", env_id=env_id, n_episodes=test_n_episode, record_video=record_video, video_dir=video_dir)

    # Q-learning — off-policy: learns optimal path along the cliff edge
    ql_agent = QLearningCliffWalkingAgent(env, alpha=0.5, epsilon=1.0, min_epsilon=0.1, gamma=1.0, decay_rate=decay_rate)
    ql_rewards = train_qlearning(ql_agent, env, n_episode=n_train)
    print(ql_agent)
    test_policy(agent=ql_agent, name="ql_agent", env_id=env_id, n_episodes=test_n_episode, record_video=record_video, video_dir=video_dir)

    plot_training_curves(
        {"SARSA": sarsa_rewards, "Q-learning": ql_rewards},
        window=10,
        save_path="videos/tabular_q/cliff_walking/training_curves.png",
    )
