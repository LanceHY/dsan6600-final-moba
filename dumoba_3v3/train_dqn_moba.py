import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from dumoba_3v5_mode.dumoba_env import DumobaEnv, MAP_OBS_N, PLAYER_OBS_N

# Try DQN Network

class DQN(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def idx_to_multidiscrete(idx: int, nvec: np.ndarray) -> np.ndarray:
    """
    Convert a flat action index into a MultiDiscrete action vector.
    """
    return np.array(np.unravel_index(idx, nvec), dtype=np.int64)


def random_action_idx(num_actions: int) -> int:
    return np.random.randint(0, num_actions)



def train_dqn(
    num_episodes: int = 500,
    max_steps_per_episode: int = 200,
    buffer_capacity: int = 100_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    initial_epsilon: float = 1.0,
    final_epsilon: float = 0.05,
    epsilon_decay_episodes: int = 300,
    target_update_interval: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    env = DumobaEnv(render_mode=None, max_steps=max_steps_per_episode)

    obs_dim = MAP_OBS_N + PLAYER_OBS_N
    nvec = env.action_space.nvec  # e.g. [7,7,3,2,2,2]
    num_actions = int(np.prod(nvec))  # 7*7*3*2*2*2 = 1176

    print(f"Observation dim: {obs_dim}, Num actions (flattened): {num_actions}")

    policy_net = DQN(obs_dim, num_actions).to(device)
    target_net = DQN(obs_dim, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    global_step = 0
    epsilon = initial_epsilon

    def get_epsilon(episode_idx: int) -> float:
        frac = min(1.0, episode_idx / epsilon_decay_episodes)
        return initial_epsilon + frac * (final_epsilon - initial_epsilon)

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0

        epsilon = get_epsilon(episode)

        for t in range(max_steps_per_episode):
            global_step += 1

            if random.random() < epsilon:
                action_idx = random_action_idx(num_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s)  # [1, num_actions]
                    action_idx = int(torch.argmax(q_values, dim=1).item())

            action_vec = idx_to_multidiscrete(action_idx, nvec)
            next_state, reward, terminated, truncated, info = env.step(action_vec)

            done = terminated or truncated
            episode_reward += reward

            replay_buffer.push(state, action_idx, reward, next_state, done)

            state = next_state

            # Update DQN
            if len(replay_buffer) >= batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                    replay_buffer.sample(batch_size)

                batch_states = torch.tensor(batch_states, dtype=torch.float32, device=device)
                batch_actions = torch.tensor(batch_actions, dtype=torch.int64, device=device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float32, device=device)

                q_values = policy_net(batch_states) 
                q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # [B]

                with torch.no_grad():
                    next_q_values = target_net(batch_next_states).max(dim=1)[0]
                    targets = batch_rewards + gamma * next_q_values * (1.0 - batch_dones)

                loss = nn.functional.mse_loss(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                # Update target net
                if global_step % target_update_interval == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(replay_buffer)}"
        )

    env.close()
    torch.save(policy_net.state_dict(), "dumoba_dqn.pt")
    print("Training finished, model saved to dumoba_dqn.pt")


if __name__ == "__main__":
    train_dqn()
