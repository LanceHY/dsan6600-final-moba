# train_ppo_dumoba_3v3.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dumoba_3v5_env import Dumoba3v5Env, MAP_OBS_N, PLAYER_OBS_N, N_BLUE
env = Dumoba3v5Env(render_mode=None)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, nvec: np.ndarray):
        super().__init__()
        self.obs_dim = obs_dim
        self.nvec = nvec
        self.hidden = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # One linear head per action dimension
        self.action_heads = nn.ModuleList(
            [nn.Linear(256, int(n)) for n in nvec]
        )
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor):
        x = self.hidden(obs)
        logits = [head(x) for head in self.action_heads]
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        """
        obs: [B, obs_dim]
        Returns:
            actions: [B, A_dims]
            log_probs: [B]
            values: [B]
        """
        logits, value = self.forward(obs)
        actions = []
        log_probs = []

        for logit in logits:
            dist = torch.distributions.Categorical(logits=logit)
            a = dist.sample()
            lp = dist.log_prob(a)
            actions.append(a)
            log_probs.append(lp)

        # Stack: each [B] → [B, A_dims]
        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)

        return actions, log_probs, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        obs: [B, obs_dim]
        actions: [B, A_dims]
        Returns:
            log_probs: [B]
            entropy: [B]
            values: [B]
        """
        logits, value = self.forward(obs)
        log_probs = []
        entropies = []

        for i, logit in enumerate(logits):
            dist = torch.distributions.Categorical(logits=logit)
            a_i = actions[:, i]
            lp = dist.log_prob(a_i)
            ent = dist.entropy()
            log_probs.append(lp)
            entropies.append(ent)

        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        entropies = torch.stack(entropies, dim=1).mean(dim=1)
        return log_probs, entropies, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones: numpy arrays of shape [T]
    returns advantages and returns
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def train_ppo(
    total_steps=100_000,
    rollout_len=256,
    batch_size=64,
    epochs=4,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
):
    env = Dumoba3v5Env(render_mode=None)
    obs_dim = MAP_OBS_N + PLAYER_OBS_N
    nvec = env.action_space.nvec

    policy = PolicyValueNet(obs_dim, nvec).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    obs, info = env.reset()
    obs = obs.astype(np.float32)
    global_step = 0

    while global_step < total_steps:
        # Rollout storage
        obs_buf = []
        actions_buf = []
        logp_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []

        for _ in range(rollout_len):
            global_step += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                actions, logp, value = policy.act(obs_tensor)

            actions_np = actions.squeeze(0).cpu().numpy()
            logp_np = logp.item()
            value_np = value.item()

            next_obs, reward, terminated, truncated, info = env.step(actions_np)
            done = terminated or truncated

            obs_buf.append(obs)
            actions_buf.append(actions_np)
            logp_buf.append(logp_np)
            rewards_buf.append(reward)
            dones_buf.append(float(done))
            values_buf.append(value_np)

            obs = next_obs.astype(np.float32)

            if done:
                obs, info = env.reset()
                obs = obs.astype(np.float32)

        # Bootstrap value
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = policy.act(obs_tensor)
        last_value = last_value.item()
        values_buf.append(last_value)

        # Convert to arrays
        rewards = np.array(rewards_buf, dtype=np.float32)
        dones = np.array(dones_buf, dtype=np.float32)
        values = np.array(values_buf, dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_arr = np.array(obs_buf, dtype=np.float32)
        actions_arr = np.array(actions_buf, dtype=np.int64)
        old_logp_arr = np.array(logp_buf, dtype=np.float32)
        returns_arr = returns

        # PPO epochs
        dataset_size = rollout_len
        idxs = np.arange(dataset_size)

        for epoch in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = torch.tensor(obs_arr[mb_idx], dtype=torch.float32, device=DEVICE)
                mb_actions = torch.tensor(actions_arr[mb_idx], dtype=torch.int64, device=DEVICE)
                mb_old_logp = torch.tensor(old_logp_arr[mb_idx], dtype=torch.float32, device=DEVICE)
                mb_adv = torch.tensor(advantages[mb_idx], dtype=torch.float32, device=DEVICE)
                mb_returns = torch.tensor(returns_arr[mb_idx], dtype=torch.float32, device=DEVICE)

                # Evaluate
                new_logp, entropy, value_pred = policy.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns - value_pred).pow(2).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        # Simple logging
        avg_return = returns_arr.mean()
        print(
            f"Steps: {global_step} | Avg return (this rollout): {avg_return:.3f} | "
            f"Adv mean: {adv_mean:.3f}, Adv std: {adv_std:.3f}"
        )

    env.close()
    torch.save(policy.state_dict(), "dumoba_3v3_ppo.pt")
    print("Training finished, saved to dumoba_3v3_ppo.pt")


if __name__ == "__main__":
    train_ppo()
