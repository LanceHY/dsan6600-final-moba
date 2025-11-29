import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dumoba_3v5_env import Dumoba3v5Env, MAP_OBS_N, PLAYER_OBS_N
from train_ppo_3v5 import PolicyValueNet, DEVICE

try:
    import imageio.v2 as imageio   # for GIFs (optional)
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def build_grid_from_env(env: Dumoba3v5Env) -> np.ndarray:
    """
    Build a simple integer grid from the current env state for visualization.

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

    # Note: env.render() prints with y reversed (top is larger y),
    # but for visualization we can keep (0,0) at bottom-left or flip later.
    return grid


def save_frame(grid: np.ndarray,
               step: int,
               episode_idx: int,
               out_dir: str,
               blue_hp_sum: int,
               red_hp_sum: int,
               blue_base_hp: int,
               red_base_hp: int):
    """Save a single frame (grid image) as PNG."""
    os.makedirs(out_dir, exist_ok=True)

    # simple colormap: we define a custom color map
    # 0 empty, 1 blue base, 2 red base, 3 blue hero, 4 red hero
    cmap = np.array([
        [1.0, 1.0, 1.0],  # 0 - empty: white
        [0.4, 0.4, 1.0],  # 1 - blue base: light blue
        [1.0, 0.4, 0.4],  # 2 - red base: light red
        [0.0, 0.0, 1.0],  # 3 - blue hero: blue
        [1.0, 0.0, 0.0],  # 4 - red hero: red
    ])

    # Map integer grid -> RGB image
    img = cmap[grid]  # shape (H, W, 3)

    plt.figure(figsize=(4, 4))
    # flip vertically so it matches your text render "top row" visually
    plt.imshow(np.flipud(img), interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.title(
        f"Episode {episode_idx+1}, Step {step}\n"
        f"Blue HP: {blue_hp_sum} (Base {blue_base_hp}) | "
        f"Red HP: {red_hp_sum} (Base {red_base_hp})"
    )
    plt.tight_layout()

    fname = os.path.join(out_dir, f"ep{episode_idx+1}_step{step:03d}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def make_gif_from_frames(out_dir: str, episode_idx: int, fps: int = 2):
    """Optional: combine PNG frames from one episode into a GIF."""
    if not HAS_IMAGEIO:
        print("imageio not installed; skipping GIF creation.")
        return

    pattern = f"ep{episode_idx+1}_step"
    frame_files = sorted(
        [f for f in os.listdir(out_dir) if f.startswith(pattern) and f.endswith(".png")]
    )
    if not frame_files:
        print(f"No frames found for episode {episode_idx+1} to create GIF.")
        return

    images = []
    for fname in frame_files:
        path = os.path.join(out_dir, fname)
        images.append(imageio.imread(path))

    gif_path = os.path.join(out_dir, f"ep{episode_idx+1}.gif")
    imageio.mimsave(gif_path, images, fps=fps)
    print(f"Saved GIF for episode {episode_idx+1} -> {gif_path}")


def evaluate_and_visualize(
    model_path: str = "dumoba_3v3_ppo.pt",
    num_episodes: int = 10,
    frames_dir: str = "frames_3v5",
    make_gifs: bool = True,
):
    env = Dumoba3v5Env(render_mode=None)
    obs_dim = MAP_OBS_N + PLAYER_OBS_N
    nvec = env.action_space.nvec

    policy = PolicyValueNet(obs_dim, nvec).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy.eval()

    print(f"Loaded model from {model_path}")
    print(f"Saving frames into: {frames_dir}")

    wins = 0
    ep_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = obs.astype(np.float32)

        done = False
        truncated = False
        step = 0
        if num_episodes <= 10:
            print(f"=== Episode {ep+1} ===")

        while not (done or truncated):
            step += 1

            # Build frame BEFORE action (state at beginning of step)
            blue_hp = sum(int(env.agent_hp[i]) for i in env.blue_ids)
            red_hp = sum(int(env.agent_hp[i]) for i in env.red_ids)
            save_frame(
                grid=build_grid_from_env(env),
                step=step,
                episode_idx=ep,
                out_dir=frames_dir,
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

        print(
            f"Episode {ep+1} finished | winner={winner}, "
            f"steps={step}"
        )

        # Make GIF per episode if requested
        if make_gifs:
            make_gif_from_frames(frames_dir, ep)

    env.close()

    win_rate = wins / num_episodes if num_episodes > 0 else 0.0
    avg_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0

    print("=== Visualization Eval Summary ===")
    print(f"Episodes (visualized): {num_episodes}")
    print(f"Blue win rate: {win_rate*100:.1f}%")
    print(f"Avg episode length: {avg_len:.2f} steps")


if __name__ == "__main__":
    # Adjust num_episodes if you want more/less footage
    evaluate_and_visualize(
        model_path="dumoba_3v3_ppo.pt",
        num_episodes=10,
        frames_dir="frames_3v5",
        make_gifs=True,
    )
