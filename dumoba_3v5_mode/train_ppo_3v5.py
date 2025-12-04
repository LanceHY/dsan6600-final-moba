# train_ppo_3v5.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dumoba_env import (
    DumobaEnv,
    GRID_SIZE,
    MAP_CHANNELS,
    MAP_OBS_N,
    PLAYER_OBS_N,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === CNN Policy + Value Net ===

class PolicyValueNetCNN(nn.Module):
    def __init__(self, nvec):
        super().__init__()
        self.nvec = nvec
        self.num_actions = len(nvec)

        # CNN over map: (B, C=MAP_CHANNELS, H=GRID_SIZE, W=GRID_SIZE)
        self.conv = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_dim = 64 * GRID_SIZE * GRID_SIZE  # 64 channels, 11x11

        # Fully connected trunk: CNN + flat features
        fc_in = conv_out_dim + PLAYER_OBS_N
        hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Policy heads: one Categorical per action dimension
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden, int(n)) for n in nvec
        ])

        # Value head
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        """
        obs: (B, MAP_OBS_N + PLAYER_OBS_N) in [0, 255]
        """
        map_flat = obs[..., :MAP_OBS_N]
        feats = obs[..., MAP_OBS_N:]

        # scale to [0,1]
        map_flat = map_flat / 255.0
        feats = feats / 255.0

        B = map_flat.shape[0]
        map_tensor = map_flat.view(B, MAP_CHANNELS, GRID_SIZE, GRID_SIZE)

        h_map = self.conv(map_tensor)      # (B, 64, 11, 11)
        h_map = h_map.reshape(B, -1)       # (B, 64*11*11)

        x = torch.cat([h_map, feats], dim=-1)
        h = self.fc(x)

        logits = [head(h) for head in self.policy_heads]
        value = self.value_head(h).squeeze(-1)  # (B,)

        return logits, value

    def act(self, obs):
        """
        obs: (B, obs_dim) float32 tensor on DEVICE
        returns:
            actions: (B, num_actions) Long
            logp: (B,) float
            value: (B,) float
        """
        logits, value = self.forward(obs)
        actions = []
        logps = []

        for head_logits in logits:
            dist = torch.distributions.Categorical(logits=head_logits)
            a = dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))

        actions = torch.stack(actions, dim=-1)      # (B, num_actions)
        logp = torch.stack(logps, dim=-1).sum(-1)   # (B,)

        return actions, logp, value


# Alias for import from eval.py
PolicyValueNet = PolicyValueNetCNN


# === PPO Training ===

def train_ppo(
    difficulty: str = "medium",
    map_type: str = "classic",
    total_steps: int = 100_000,
    rollout_len: int = 256,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    batch_size: int = 64,
    epochs: int = 4,
    save_dir: str = "models",
):
    os.makedirs(save_dir, exist_ok=True)

    env = DumobaEnv(difficulty=difficulty, map_type=map_type, render_mode=None)
    nvec = env.action_space.nvec

    policy = PolicyValueNet(nvec).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    obs, _ = env.reset()
    obs = obs.astype(np.float32)

    step_count = 0
    update_idx = 0

    while step_count < total_steps:
        # Collect rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        for _ in range(rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                actions, logp, value = policy.act(obs_t)

            actions_np = actions.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(actions_np)
            next_obs = next_obs.astype(np.float32)

            obs_buf.append(obs)
            act_buf.append(actions_np)
            logp_buf.append(logp.item())
            val_buf.append(value.item())
            rew_buf.append(reward)
            done_buf.append(done)

            obs = next_obs
            step_count += 1

            if done or truncated:
                obs, _ = env.reset()
                obs = obs.astype(np.float32)

            if step_count >= total_steps:
                break

        # Convert rollout to arrays
        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.int64)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)

        # Compute GAE advantages and returns
        T_len = len(rew_arr)
        advs = np.zeros_like(rew_arr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T_len)):
            next_value = val_arr[t + 1] if t + 1 < T_len else 0.0
            delta = rew_arr[t] + gamma * next_value * (1 - done_arr[t]) - val_arr[t]
            last_gae = delta + gamma * gae_lambda * (1 - done_arr[t]) * last_gae
            advs[t] = last_gae

        returns = advs + val_arr

        # Normalize advantages
        advs_mean = advs.mean()
        advs_std = advs.std() + 1e-8
        advs = (advs - advs_mean) / advs_std

        # Convert to tensors
        obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=DEVICE)
        act_tensor = torch.tensor(act_arr, dtype=torch.long, device=DEVICE)
        old_logp_tensor = torch.tensor(logp_arr, dtype=torch.float32, device=DEVICE)
        adv_tensor = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # PPO updates
        T_idx = np.arange(T_len)
        for _ in range(epochs):
            np.random.shuffle(T_idx)
            for start in range(0, T_len, batch_size):
                idx = T_idx[start:start + batch_size]
                batch_obs = obs_tensor[idx]
                batch_act = act_tensor[idx]
                batch_old_logp = old_logp_tensor[idx]
                batch_adv = adv_tensor[idx]
                batch_ret = ret_tensor[idx]

                logits, value = policy.forward(batch_obs)

                # recompute logprob
                logps_list = []
                for i, head_logits in enumerate(logits):
                    dist = torch.distributions.Categorical(logits=head_logits)
                    logps_list.append(dist.log_prob(batch_act[:, i]))

                new_logp = torch.stack(logps_list, dim=-1).sum(-1)

                ratio = torch.exp(new_logp - batch_old_logp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((value - batch_ret) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        update_idx += 1
        print(
            f"[Train] Update {update_idx}, Steps {step_count}/{total_steps}, "
            f"Loss {loss.item():.4f}"
        )

    # Save final model
    model_name = f"dumoba_3v5_{difficulty}_{map_type}.pt"
    save_path = os.path.join(save_dir, model_name)
    torch.save(policy.state_dict(), save_path)
    print(f"Saved model -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--map-type", type=str, default="classic",
                        choices=["classic", "corner"])
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    train_ppo(
        difficulty=args.difficulty,
        map_type=args.map_type,
        total_steps=args.total_steps,
        save_dir=args.save_dir,
    )
