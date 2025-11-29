import numpy as np
import torch

from dumoba_3v5_env import Dumoba3v5Env, MAP_OBS_N, PLAYER_OBS_N
from train_ppo_3v5 import PolicyValueNet, DEVICE  # or whatever your train file is called


def evaluate_3v5(
    model_path: str = "dumoba_3v3_ppo.pt",
    num_episodes: int = 100,
    render: bool = False,
):
    env = Dumoba3v5Env(render_mode="human" if render else None)
    obs_dim = MAP_OBS_N + PLAYER_OBS_N
    nvec = env.action_space.nvec

    policy = PolicyValueNet(obs_dim, nvec).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy.eval()

    wins = 0
    ep_lengths = []
    blue_hp_totals = []
    red_hp_totals = []
    blue_base_hp_totals = []
    red_base_hp_totals = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = obs.astype(np.float32)

        done = False
        truncated = False
        ep_return = 0.0
        steps = 0

        if render:
            print(f"=== Episode {ep+1} ===")

        while not (done or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                actions, logp, value = policy.act(obs_tensor)

            actions_np = actions.squeeze(0).cpu().numpy()
            obs, reward, done, truncated, info = env.step(actions_np)
            obs = obs.astype(np.float32)
            ep_return += reward
            steps += 1

        # Episode finished
        winner = info.get("winner", None)
        if winner == "blue":
            wins += 1

        ep_lengths.append(steps)

        # Grab final HP stats from env
        blue_hp = sum(int(env.agent_hp[i]) for i in env.blue_ids)
        red_hp = sum(int(env.agent_hp[i]) for i in env.red_ids)
        blue_hp_totals.append(blue_hp)
        red_hp_totals.append(red_hp)
        blue_base_hp_totals.append(int(env.blue_base_hp))
        red_base_hp_totals.append(int(env.red_base_hp))

        if render:
            print(
                f"Episode {ep+1} done | winner={winner}, "
                f"steps={steps}, return={ep_return:.3f}, "
                f"blue_hp={blue_hp}, red_hp={red_hp}"
            )

    env.close()

    win_rate = wins / num_episodes
    avg_len = float(np.mean(ep_lengths))
    avg_blue_hp = float(np.mean(blue_hp_totals))
    avg_red_hp = float(np.mean(red_hp_totals))
    avg_blue_base_hp = float(np.mean(blue_base_hp_totals))
    avg_red_base_hp = float(np.mean(red_base_hp_totals))

    print("=== Dumoba 3v5 Evaluation Summary ===")
    print(f"Model:          {model_path}")
    print(f"Episodes:       {num_episodes}")
    print(f"Win rate (blue): {win_rate*100:.1f}%")
    print(f"Avg ep length:   {avg_len:.2f} steps")
    print(f"Avg blue hero HP sum: {avg_blue_hp:.2f}")
    print(f"Avg red hero HP sum:  {avg_red_hp:.2f}")
    print(f"Avg blue base HP:     {avg_blue_base_hp:.2f}")
    print(f"Avg red base HP:      {avg_red_base_hp:.2f}")

    # Return stats in case you want to write them to a file
    return {
        "win_rate": win_rate,
        "avg_ep_len": avg_len,
        "avg_blue_hp": avg_blue_hp,
        "avg_red_hp": avg_red_hp,
        "avg_blue_base_hp": avg_blue_base_hp,
        "avg_red_base_hp": avg_red_base_hp,
    }


if __name__ == "__main__":
    # Turn render=True if you want to visually inspect a few games
    stats = evaluate_3v5(model_path="dumoba_3v3_ppo.pt", num_episodes=100, render=False)
