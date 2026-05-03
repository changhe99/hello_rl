import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod
from gymnasium.spaces.utils import flatdim


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


class QLFrozenLakeAgent(FrozenLakeAgent):
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

    def update(self, state, action, reward, next_state):
        if next_state is None:
            self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.alpha * (reward + self.gamma * max(self.q_table[next_state]) - self.q_table[state, action])

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


def run_policy(agent: FrozenLakeAgent, name: str, env: Env, n_episodes: int = 10, record_video: bool = True, video_dir: str = "videos/tabular_q/frozen_lake/") -> None:
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
    print(f"Win rate: {total_rewards / n_episodes:.1%}")

def train_ql(agent: QLFrozenLakeAgent, env: Env, n_episode: int = 1000):
    with tqdm(range(n_episode), desc="training") as pbar:
        for ep in pbar:
            obs, _ = env.reset()
            done = False

            while not done:
                prev_obs = obs
                action = agent.act(obs, eval=False)
                obs, rew, ter, tru, _ = env.step(action)
                done = ter or tru
                agent.update(prev_obs, action, rew, None if ter else obs)

            pbar.set_postfix(epsilon=f"{agent.epsilon:.4f}")

if __name__ == "__main__":
    video_dir = "videos/tabular_q/frozen_lake/"
    record_video = True
    n_episode = 10

    # Env param
    slippery = True

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=slippery, success_rate=3.0/4.0, reward_schedule=(1, 0, 0), render_mode="rgb_array" if record_video else None)

    # random agent
    random_agent = RandomFrozenLakeAgent(env)
    run_policy(agent=random_agent, name="random_agent", env=env, n_episodes=n_episode, record_video=record_video, video_dir=video_dir)

    # QL agent
    ql_agent = QLFrozenLakeAgent(env, alpha=0.2, epsilon=1.0, min_epsilon=0.1, gamma=0.99, decay_rate=0.999999)
    train_ql(ql_agent, env, n_episode=800000)
    run_policy(agent=ql_agent, name="ql_agent", env=env, n_episodes=n_episode, record_video=record_video, video_dir=video_dir)
