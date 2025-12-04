# eval.py

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from dumoba_env import DumobaEnv, GRID_SIZE
from train_ppo_3v5 import PolicyValueNet, DEVICE


try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def build_grid_from_env(env: DumobaEnv) -> np.ndarray:
    """
    Build a simple integer grid from current env state.

    Codes:
        0 = empty
        1 = blue base
        2 = red base
        3 = blue hero
        4 = red hero
    """
    gsize = env.grid_size
    grid = np.zeros((gsize, gsize), dtype=np.int32)

    # Bases
    bx, by = env.blue_base_pos
    if env.blue_base_hp > 0:
        grid[by, bx] = 1

    rx, ry = env.red_base_pos
    if env.red_base_hp > 0:
        grid[ry, rx] = 2

    # Agents
    for i in range(env.n_agents):
        if env.agent_hp[i] <= 0:
            continue
        x, y = env.agent_positions[i]
        if i in env.blue_ids:
            grid[y, x] = 3
        else:
            grid[y, x] = 4

    return grid


def save_frame(
    grid: np.ndarray,
    step: int,
    episode_idx: int,
    out_dir: str,
    difficulty: str,
    blue_hp_sum: int,
    red_hp_sum: int,
    blue_base_hp: int,
    red_base_hp: int,
):
    """Save a single frame (grid image) as PNG."""
    os.makedirs(out_dir, exist_ok=True)

    # simple colormap
    cmap = np.array([
        [1.0, 1.0, 1.0],  # 0 - empty: white
        [0.4, 0.4, 1.0],  # 1 - blue base
        [1.0, 0.4, 0.4],  # 2 - red base
        [0.0, 0.0, 1.0],  # 3 - blue hero
        [1.0, 0.0, 0.0],  # 4 - red hero
    ])

    img = cmap[grid]

    plt.figure(figsize=(4, 4))
    plt.imshow(np.flipud(img), interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.title(
        f"{difficulty.capitalize()} | Ep {episode_idx+1}, Step {step}\n"
        f"Blue HP: {blue_hp_sum} (Base {blue_base_hp}) | "
        f"Red HP: {red_hp_sum} (Base {red_base_hp})"
    )
    plt.tight_layout()

    fname = os.path.join(
        out_dir, f"{difficulty}_ep{episode_idx+1}_step{step:03d}.png"
    )
    plt.savefig(fname, dpi=150)
    plt.close()


def make_gif_from_frames(out_dir: str, difficulty: str, episode_idx: int, fps: int = 2):
    """Optional: combine PNG frames from one episode into a GIF."""
    if not HAS_IMAGEIO:
        print("imageio not installed; skipping GIF creation.")
        return

    pattern = f"{difficulty}_ep{episode_idx+1}_step"
    frame_files = sorted(
        [f for f in os.listdir(out_dir) if f.startswith(pattern) and f.endswith(".png")]
    )
    if not frame_files:
        print(f"No frames found for {difficulty}, episode {episode_idx+1}")
        return

    images = []
    for fname in frame_files:
        path = os.path.join(out_dir, fname)
        images.append(imageio.imread(path))

    gif_path = os.path.join(out_dir, f"{difficulty}_ep{episode_idx+1}.gif")
    imageio.mimsave(gif_path, images, fps=fps)
    print(f"Saved GIF -> {gif_path}")


def evaluate_mode(
    difficulty: str,
    map_type: str = "classic",
    model_dir: str = "models",
    results_root: str = "results",
    num_episodes_eval: int = 100,
    num_episodes_visual: int = 3,
):
    """
    Evaluate a trained model for one difficulty mode and one map type:
    - Computes win rate, avg episode length, HP stats over num_episodes_eval.
    - Saves frames + GIFs for first num_episodes_visual episodes.
    - Writes a JSON + text summary to results/{map_type}/{difficulty}/.
    """
    model_name = f"dumoba_3v5_{difficulty}_{map_type}.pt"
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found for {difficulty}: {model_path} (skipping)")
        return

    out_dir = os.path.join(results_root, map_type, difficulty)
    os.makedirs(out_dir, exist_ok=True)

    env = DumobaEnv(difficulty=difficulty, map_type=map_type, render_mode=None)
    nvec = env.action_space.nvec

    policy = PolicyValueNet(nvec).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy.eval()

    wins = 0
    ep_lengths = []
    blue_hp_sums = []
    red_hp_sums = []
    blue_base_hps = []
    red_base_hps = []

    for ep in range(num_episodes_eval):
        obs, info = env.reset()
        obs = obs.astype(np.float32)

        done = False
        truncated = False
        step = 0
        winner = None

        frames_for_this_ep = (ep < num_episodes_visual)

        while not (done or truncated):
            step += 1

            if frames_for_this_ep:
                blue_hp = sum(int(env.agent_hp[i]) for i in env.blue_ids)
                red_hp = sum(int(env.agent_hp[i]) for i in env.red_ids)
                save_frame(
                    grid=build_grid_from_env(env),
                    step=step,
                    episode_idx=ep,
                    out_dir=out_dir,
                    difficulty=difficulty,
                    blue_hp_sum=blue_hp,
                    red_hp_sum=red_hp,
                    blue_base_hp=int(env.blue_base_hp),
                    red_base_hp=int(env.red_base_hp),
                )

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                actions, logp, value = policy.act(obs_t)

            actions_np = actions.squeeze(0).cpu().numpy()
            obs, reward, done, truncated, info = env.step(actions_np)
            obs = obs.astype(np.float32)
            winner = info.get("winner", None)

        if winner == "blue":
            wins += 1

        ep_lengths.append(step)
        blue_hp_sums.append(sum(int(env.agent_hp[i]) for i in env.blue_ids))
        red_hp_sums.append(sum(int(env.agent_hp[i]) for i in env.red_ids))
        blue_base_hps.append(int(env.blue_base_hp))
        red_base_hps.append(int(env.red_base_hp))

        print(
            f"[{difficulty}] Episode {ep+1}/{num_episodes_eval} "
            f"| winner={winner}, steps={step}"
        )

        if frames_for_this_ep:
            make_gif_from_frames(out_dir, difficulty, ep)

    env.close()

    win_rate = wins / num_episodes_eval if num_episodes_eval > 0 else 0.0
    avg_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    avg_blue_hp = float(np.mean(blue_hp_sums)) if blue_hp_sums else 0.0
    avg_red_hp = float(np.mean(red_hp_sums)) if red_hp_sums else 0.0
    avg_blue_base_hp = float(np.mean(blue_base_hps)) if blue_base_hps else 0.0
    avg_red_base_hp = float(np.mean(red_base_hps)) if red_base_hps else 0.0

    summary = {
        "difficulty": difficulty,
        "map_type": map_type,
        "episodes": num_episodes_eval,
        "win_rate_blue": win_rate,
        "avg_episode_length": avg_len,
        "avg_blue_hero_hp_sum": avg_blue_hp,
        "avg_red_hero_hp_sum": avg_red_hp,
        "avg_blue_base_hp": avg_blue_base_hp,
        "avg_red_base_hp": avg_red_base_hp,
    }

    # Save JSON summary
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save a human-readable text summary
    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== Dumoba 3v5 Evaluation Summary ===\n")
        f.write(f"Difficulty:      {difficulty}\n")
        f.write(f"Map type:        {map_type}\n")
        f.write(f"Episodes:        {num_episodes_eval}\n")
        f.write(f"Win rate (blue): {win_rate * 100:.1f}%\n")
        f.write(f"Avg ep length:   {avg_len:.2f} steps\n")
        f.write(f"Avg blue hero HP sum: {avg_blue_hp:.2f}\n")
        f.write(f"Avg red hero  HP sum: {avg_red_hp:.2f}\n")
        f.write(f"Avg blue base HP:     {avg_blue_base_hp:.2f}\n")
        f.write(f"Avg red base  HP:     {avg_red_base_hp:.2f}\n")

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    print(f"Saved summary to {txt_path}")
    print(f"Frames/GIFs in: {out_dir}")


if __name__ == "__main__":
    # You can adjust map_type if you like: "classic" or "corner"
    map_type = "classic"

    # Evaluate all three difficulty modes
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n=== Evaluating difficulty: {difficulty} ===")
        evaluate_mode(
            difficulty=difficulty,
            map_type=map_type,
            model_dir="models",
            results_root="results",
            num_episodes_eval=100,
            num_episodes_visual=3,
        )
