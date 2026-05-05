"""
Train SARSA and Q-learning on CliffWalking and produce GitHub-ready demo assets:
  docs/tabular_q/cliff_walking/sarsa.gif
  docs/tabular_q/cliff_walking/ql.gif
  docs/tabular_q/cliff_walking/training_curves.png
  docs/tabular_q/cliff_walking/CLIFF_DEMO.md

Run from repo root:
  python -m hello_rl.tabular_q.make_cliff_demo
"""

import os
import textwrap

import gymnasium as gym
import imageio
import numpy as np

from hello_rl.tabular_q.cliff_walking import (
    QLearningCliffWalkingAgent,
    SARSACliffWalkingAgent,
    plot_training_curves,
    train_qlearning,
    train_sarsa,
)

OUT_DIR = "docs/tabular_q/cliff_walking"
ENV_ID = "CliffWalking-v1"
MAX_STEPS = 200
N_TRAIN = 5000
# decay epsilon 1.0 → 0.1 over N_TRAIN episodes × ~20 steps/ep
DECAY_RATE = 0.1 ** (1 / (N_TRAIN * 20))


def record_episode(agent, fps: int = 8) -> list[np.ndarray]:
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    obs, _ = env.reset()
    frames = [env.render()]
    done = False
    while not done:
        action = agent.act(obs, eval=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        done = terminated or truncated
    env.close()
    return frames


def save_gif(frames: list[np.ndarray], path: str, fps: int = 8) -> None:
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  saved {path}  ({len(frames)} frames)")


def write_demo_md(out_dir: str, sarsa_repr: str, ql_repr: str) -> None:
    sarsa_grid = "\n".join(f"    {line}" for line in sarsa_repr.splitlines())
    ql_grid = "\n".join(f"    {line}" for line in ql_repr.splitlines())

    content = textwrap.dedent(f"""\
        # CliffWalking — SARSA vs Q-learning

        CliffWalking is a 4×12 grid with a cliff along the bottom edge.
        Each step costs −1; falling off the cliff costs −100 and resets to start.
        The goal is to reach the bottom-right corner in as few steps as possible.

        This is the classic environment from **Sutton & Barto Chapter 6** that
        highlights the on-policy vs off-policy distinction:

        | | SARSA (on-policy) | Q-learning (off-policy) |
        |---|---|---|
        | **Update target** | Q(s′, a′) — next action from ε-greedy | max Q(s′, ·) — greedy best |
        | **Learned path** | Safe path one row above cliff | Optimal cliff-edge path |
        | **Eval reward** | ~−17 (longer but safe) | ~−13 (shortest path) |
        | **Training noise** | Low (avoids cliff during exploration) | High (falls off cliff while ε > 0) |

        ## Training curves

        SARSA converges to a higher (safer) online reward during training because
        its on-policy update accounts for accidental cliff falls under ε-greedy.
        Q-learning has a noisier training curve but discovers the shorter path.

        ![training curves](training_curves.png)

        ## Learned policies (one eval episode each)

        <table>
        <tr>
          <th>SARSA — safe path (~−17)</th>
          <th>Q-learning — cliff-edge path (~−13)</th>
        </tr>
        <tr>
          <td><img src="sarsa.gif" alt="SARSA episode"/></td>
          <td><img src="ql.gif" alt="Q-learning episode"/></td>
        </tr>
        </table>

        ## Policy grids

        `S` = start, `G` = goal, `C` = cliff.
        Actions: `U` up · `R` right · `D` down · `L` left.

        **SARSA**
        ```
    {sarsa_grid}
        ```

        **Q-learning**
        ```
    {ql_grid}
        ```
    """)

    path = os.path.join(out_dir, "CLIFF_DEMO.md")
    with open(path, "w") as f:
        f.write(content)
    print(f"  saved {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    env = gym.make(ENV_ID, max_episode_steps=MAX_STEPS)

    print("Training SARSA …")
    sarsa_agent = SARSACliffWalkingAgent(
        env, alpha=0.5, epsilon=1.0, min_epsilon=0.1, gamma=1.0, decay_rate=DECAY_RATE
    )
    sarsa_rewards = train_sarsa(sarsa_agent, env, n_episode=N_TRAIN)

    print("Training Q-learning …")
    ql_agent = QLearningCliffWalkingAgent(
        env, alpha=0.5, epsilon=1.0, min_epsilon=0.1, gamma=1.0, decay_rate=DECAY_RATE
    )
    ql_rewards = train_qlearning(ql_agent, env, n_episode=N_TRAIN)

    env.close()

    print("Saving training curves …")
    plot_training_curves(
        {"SARSA": sarsa_rewards, "Q-learning": ql_rewards},
        window=50,
        save_path=os.path.join(OUT_DIR, "training_curves.png"),
    )

    print("Recording SARSA episode …")
    save_gif(record_episode(sarsa_agent), os.path.join(OUT_DIR, "sarsa.gif"))

    print("Recording Q-learning episode …")
    save_gif(record_episode(ql_agent), os.path.join(OUT_DIR, "ql.gif"))

    print("Writing CLIFF_DEMO.md …")
    write_demo_md(OUT_DIR, repr(sarsa_agent), repr(ql_agent))

    print("\nDone. Commit docs/tabular_q/cliff_walking/ and open CLIFF_DEMO.md on GitHub.")


if __name__ == "__main__":
    main()
