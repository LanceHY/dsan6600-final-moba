# dumoba_3v3_env.py

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

GRID_SIZE = 11
N_BLUE = 3
N_RED = 3
N_AGENTS = N_BLUE + N_RED

MAP_CHANNELS = 8  # 3 blue heroes, 3 red heroes, 2 bases
MAP_OBS_N = GRID_SIZE * GRID_SIZE * MAP_CHANNELS
PLAYER_OBS_N = 32  # features for heroes + bases, padded


class Dumoba3v3Env(gym.Env):
    """
    Dumoba 3v3: A simple MOBA-like 3v3 environment.

    - 3 blue heroes vs 3 red heroes on an 11x11 grid.
    - Single RL "meta-agent" controls all 3 blue heroes jointly.
    - Red heroes are scripted opponents.
    - Win by destroying all red heroes or their base.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode: str | None = None, max_steps: int = 200):
        super().__init__()

        self.render_mode = render_mode
        self.grid_size = GRID_SIZE
        self.max_steps = max_steps

        # Agent indexing
        self.blue_ids = np.array([0, 1, 2], dtype=np.int32)
        self.red_ids = np.array([3, 4, 5], dtype=np.int32)
        self.n_agents = N_AGENTS

        # HP config
        self.max_hero_hp = 10
        self.max_base_hp = 20
        self.attack_damage = 2

        # Bases: blue at top center, red at bottom center (y=grid-1 is top when printed)
        self.blue_base_pos = np.array([self.grid_size // 2, self.grid_size - 1], dtype=np.int32)
        self.red_base_pos = np.array([self.grid_size // 2, 0], dtype=np.int32)

        # Observation: flattened map + feature vector, uint8
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(MAP_OBS_N + PLAYER_OBS_N,),
            dtype=np.uint8,
        )

        # Action: 3 heroes * MultiDiscrete([7,7,3,2,2,2]) → 18 dims
        single_hero_nvec = np.array([7, 7, 3, 2, 2, 2], dtype=np.int64)
        self.nvec_single = single_hero_nvec
        self.nvec = np.tile(single_hero_nvec, N_BLUE)  # shape (18,)
        self.action_space = spaces.MultiDiscrete(self.nvec)

        # Internal state
        self.agent_positions = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.agent_hp = np.zeros((self.n_agents,), dtype=np.int32)
        self.blue_base_hp = None
        self.red_base_hp = None
        self.steps = 0

    # ------------- Gym API ------------- #

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # Spawn blue heroes near blue base
        # blue base: (cx, top row), heroes one row below
        cx = self.blue_base_pos[0]
        top_row = self.blue_base_pos[1]
        blue_row = max(0, top_row - 1)
        blue_xs = [cx - 1, cx, cx + 1]

        # Spawn red heroes near red base
        cx_r = self.red_base_pos[0]
        bottom_row = self.red_base_pos[1]
        red_row = min(self.grid_size - 1, bottom_row + 1)
        red_xs = [cx_r - 1, cx_r, cx_r + 1]

        # Clamp x positions to grid
        blue_xs = [int(np.clip(x, 0, self.grid_size - 1)) for x in blue_xs]
        red_xs = [int(np.clip(x, 0, self.grid_size - 1)) for x in red_xs]

        # Set positions
        for i, x in enumerate(blue_xs):
            self.agent_positions[self.blue_ids[i]] = np.array([x, blue_row], dtype=np.int32)

        for i, x in enumerate(red_xs):
            self.agent_positions[self.red_ids[i]] = np.array([x, red_row], dtype=np.int32)

        # HP
        self.agent_hp[:] = self.max_hero_hp
        self.blue_base_hp = self.max_base_hp
        self.red_base_hp = self.max_base_hp
        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        action: MultiDiscrete(18,) → 3 heroes × 6 dims each
        """
        self.steps += 1
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (N_BLUE * 6,):
            raise ValueError(f"Expected action shape {(N_BLUE*6,)}, got {action.shape}")

        # Split action into per-hero actions
        hero_actions = action.reshape(N_BLUE, 6)

        # 1) Apply movement for blue heroes
        for i, hero_id in enumerate(self.blue_ids):
            if self._is_alive(hero_id):
                self._apply_movement(hero_id, int(hero_actions[i][0]))

        # 2) Apply movement for red heroes (scripted)
        red_actions = []
        for red_id in self.red_ids:
            if self._is_alive(red_id):
                red_action = self._scripted_policy(red_id)
            else:
                # dead: do nothing
                red_action = np.zeros(6, dtype=np.int64)
            red_actions.append(red_action)

        red_actions = np.asarray(red_actions, dtype=np.int64)

        for i, red_id in enumerate(self.red_ids):
            if self._is_alive(red_id):
                self._apply_movement(red_id, int(red_actions[i][0]))

        # 3) Resolve attacks (blue, then red)
        # Blue heroes attack
        for i, hero_id in enumerate(self.blue_ids):
            if self._is_alive(hero_id):
                self._apply_attack(hero_id, hero_actions[i])

        # Red heroes attack
        for i, red_id in enumerate(self.red_ids):
            if self._is_alive(red_id):
                self._apply_attack(red_id, red_actions[i])

        # 4) Check terminal condition
        terminated, winner = self._check_done()

        # 5) Compute reward
        reward = self._compute_reward(terminated, winner)

        truncated = False
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {"winner": winner}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Bases
        bx, by = self.blue_base_pos
        rx, ry = self.red_base_pos
        if self.blue_base_hp > 0:
            grid[by][bx] = "B"
        if self.red_base_hp > 0:
            grid[ry][rx] = "b"

        # Agents
        for i in range(self.n_agents):
            if self.agent_hp[i] <= 0:
                continue
            x, y = self.agent_positions[i]
            if i in self.blue_ids:
                label = f"{i}"  # 0,1,2
            else:
                label = f"r{i-3}"  # r0,r1,r2
            grid[y][x] = label

        print(f"Step {self.steps}")
        # Print top row last so visually top of map is top
        for row in reversed(grid):
            print(" ".join(row))
        print(
            f"Blue HP: {[int(self.agent_hp[i]) for i in self.blue_ids]}, "
            f"Blue Base: {self.blue_base_hp} | "
            f"Red HP: {[int(self.agent_hp[i]) for i in self.red_ids]}, "
            f"Red Base: {self.red_base_hp}"
        )
        print()

    def close(self):
        pass

    # ------------- Helpers ------------- #

    def _is_alive(self, agent_id: int) -> bool:
        return self.agent_hp[agent_id] > 0

    def _in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _move(self, pos, direction: int):
        # 0 = stay; 1 = up; 2 = down; 3 = left; 4 = right; others = stay
        x, y = pos
        if direction == 1:
            y += 1
        elif direction == 2:
            y -= 1
        elif direction == 3:
            x -= 1
        elif direction == 4:
            x += 1

        new_pos = np.array([x, y], dtype=np.int32)
        if self._in_bounds(new_pos):
            return new_pos
        return pos

    def _apply_movement(self, agent_id: int, move_dir: int):
        if move_dir not in [1, 2, 3, 4]:
            return
        self.agent_positions[agent_id] = self._move(self.agent_positions[agent_id], move_dir)

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _attack_one_target(self, attacker_id: int, target_pos, target_hp: int):
        """Attack target if in range 1."""
        if target_hp <= 0:
            return target_hp
        if self._manhattan(self.agent_positions[attacker_id], target_pos) <= 1:
            return max(0, target_hp - self.attack_damage)
        return target_hp

    def _apply_attack(self, agent_id: int, action_vec: np.ndarray):
        """
        action_vec: length-6 MultiDiscrete for this agent.
        We only use:
          - action_vec[3] = attack_flag (0/1)
        For simplicity we attack:
          - any enemy hero in range (first found)
          - then enemy base if in range
        """
        attack_flag = int(action_vec[3])
        if attack_flag != 1:
            return

        # Decide which team we're on
        if agent_id in self.blue_ids:
            enemy_ids = self.red_ids
            enemy_base_pos = self.red_base_pos
            base_hp_attr = "red_base_hp"
        else:
            enemy_ids = self.blue_ids
            enemy_base_pos = self.blue_base_pos
            base_hp_attr = "blue_base_hp"

        # Attack enemy hero if any in range
        for eid in enemy_ids:
            if self.agent_hp[eid] > 0:
                self.agent_hp[eid] = self._attack_one_target(
                    agent_id, self.agent_positions[eid], self.agent_hp[eid]
                )
                if self.agent_hp[eid] <= 0:
                    # killed someone; you could add bonus reward per kill if you want
                    pass

        # Attack enemy base if in range
        base_hp = getattr(self, base_hp_attr)
        base_hp = self._attack_one_target(agent_id, enemy_base_pos, base_hp)
        setattr(self, base_hp_attr, base_hp)

    def _nearest_enemy(self, agent_id: int):
        """Return position of nearest enemy for this agent."""
        if agent_id in self.blue_ids:
            enemy_ids = self.red_ids
        else:
            enemy_ids = self.blue_ids

        my_pos = self.agent_positions[agent_id]
        best_dist = 9999
        best_pos = None

        for eid in enemy_ids:
            if self.agent_hp[eid] <= 0:
                continue
            pos = self.agent_positions[eid]
            d = self._manhattan(my_pos, pos)
            if d < best_dist:
                best_dist = d
                best_pos = pos

        # If no enemy alive, chase enemy base
        if best_pos is None:
            if agent_id in self.blue_ids:
                best_pos = self.red_base_pos
            else:
                best_pos = self.blue_base_pos

        return best_pos

    def _scripted_policy(self, agent_id: int) -> np.ndarray:
        """
        Very simple scripted policy:
          - Move greedily towards nearest enemy or enemy base
          - Attack if in range
        Returns a length-6 action vector.
        """
        action = np.zeros(6, dtype=np.int64)
        if not self._is_alive(agent_id):
            return action

        # Movement
        target_pos = self._nearest_enemy(agent_id)
        my_pos = self.agent_positions[agent_id]

        dx = np.sign(target_pos[0] - my_pos[0])
        dy = np.sign(target_pos[1] - my_pos[1])

        # Use same encoding as _move
        if dy > 0:
            move_dir = 1  # up
        elif dy < 0:
            move_dir = 2  # down
        elif dx < 0:
            move_dir = 3  # left
        elif dx > 0:
            move_dir = 4  # right
        else:
            move_dir = 0  # stay

        action[0] = move_dir

        # Attack if in range of any enemy or enemy base
        # Attack flag at index 3
        in_range = False
        if agent_id in self.blue_ids:
            enemy_ids = self.red_ids
            base_pos = self.red_base_pos
        else:
            enemy_ids = self.blue_ids
            base_pos = self.blue_base_pos

        for eid in enemy_ids:
            if self.agent_hp[eid] > 0:
                if self._manhattan(my_pos, self.agent_positions[eid]) <= 1:
                    in_range = True
                    break

        if not in_range and self._manhattan(my_pos, base_pos) <= 1:
            in_range = True

        action[3] = 1 if in_range else 0
        return action

    def _check_done(self):
        """
        Check if game is over.
        Winner: "blue", "red", or None.
        """
        blue_alive = any(self.agent_hp[i] > 0 for i in self.blue_ids)
        red_alive = any(self.agent_hp[i] > 0 for i in self.red_ids)

        # Base destroyed or all heroes dead
        if (self.blue_base_hp <= 0) or (not blue_alive):
            return True, "red"
        if (self.red_base_hp <= 0) or (not red_alive):
            return True, "blue"

        return False, None

    def _compute_reward(self, terminated: bool, winner):
        if not terminated:
            # Small time penalty to encourage faster resolutions
            return -0.01

        if winner == "blue":
            return 1.0
        elif winner == "red":
            return -1.0
        return 0.0

    def _get_obs(self) -> np.ndarray:
        """
        Returns a uint8 vector of length MAP_OBS_N + PLAYER_OBS_N:
          - 11x11x8 map
          - + 32-dim feature vector (positions/HP for all heroes + base HPs)
        """
        # Map channels:
        # 0,1,2 -> blue heroes 0,1,2
        # 3,4,5 -> red heroes 3,4,5
        # 6 -> blue base
        # 7 -> red base
        map_grid = np.zeros((self.grid_size, self.grid_size, MAP_CHANNELS), dtype=np.uint8)

        # Heroes
        for idx, agent_id in enumerate(self.blue_ids):
            if self.agent_hp[agent_id] > 0:
                x, y = self.agent_positions[agent_id]
                map_grid[y, x, idx] = 1

        for idx, agent_id in enumerate(self.red_ids):
            if self.agent_hp[agent_id] > 0:
                x, y = self.agent_positions[agent_id]
                map_grid[y, x, 3 + idx] = 1

        # Bases
        if self.blue_base_hp > 0:
            bx, by = self.blue_base_pos
            map_grid[by, bx, 6] = 1
        if self.red_base_hp > 0:
            rx, ry = self.red_base_pos
            map_grid[ry, rx, 7] = 1

        map_flat = map_grid.flatten()

        # Features for all heroes (x, y, hp) + base hp
        feats = []
        for agent_id in range(self.n_agents):
            feats.append(self.agent_positions[agent_id][0])
            feats.append(self.agent_positions[agent_id][1])
            feats.append(self.agent_hp[agent_id])

        feats.append(self.blue_base_hp)
        feats.append(self.red_base_hp)

        # Pad to PLAYER_OBS_N
        if len(feats) < PLAYER_OBS_N:
            feats.extend([0] * (PLAYER_OBS_N - len(feats)))
        elif len(feats) > PLAYER_OBS_N:
            feats = feats[:PLAYER_OBS_N]

        feats_arr = np.array(feats, dtype=np.uint8)
        obs = np.concatenate([map_flat, feats_arr], dtype=np.uint8)
        return obs


if __name__ == "__main__":
    env = Dumoba3v3Env(render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        a = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(a)
        if done or truncated:
            print("Episode finished. Reward:", reward, "info:", info)
            break
