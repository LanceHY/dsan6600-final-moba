# eval_ppo_dumoba_3v3.py

import numpy as np
import torch

from dumoba_3v3_env import Dumoba3v3Env, MAP_OBS_N, PLAYER_OBS_N
from train_ppo_3v3 import PolicyValueNet, DEVICE


def evaluate(num_episodes=5):
    env = Dumoba3v3Env(render_mode="human")
    obs_dim = MAP_OBS_N + PLAYER_OBS_N
    nvec = env.action_space.nvec

    policy = PolicyValueNet(obs_dim, nvec).to(DEVICE)
    policy.load_state_dict(torch.load("dumoba_3v3_ppo.pt", map_location=DEVICE))
    policy.eval()

    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = obs.astype(np.float32)
        done = False
        truncated = False
        ep_return = 0.0

        print(f"=== Episode {ep+1} ===")
        while not (done or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                actions, logp, value = policy.act(obs_tensor)

            actions_np = actions.squeeze(0).cpu().numpy()
            obs, reward, done, truncated, info = env.step(actions_np)
            obs = obs.astype(np.float32)
            ep_return += reward

        print(f"Episode {ep+1} finished. Return: {ep_return:.3f}, info: {info}")

    env.close()


if __name__ == "__main__":
    evaluate()
