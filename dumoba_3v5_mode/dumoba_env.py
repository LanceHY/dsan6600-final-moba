# dumoba_env.py

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# === Global constants ===

GRID_SIZE = 11
N_BLUE = 3
N_RED = 5
N_AGENTS = N_BLUE + N_RED

# Channels:
# 0-2: blue heroes
# 3-7: red heroes
# 8: blue base
# 9: red base
MAP_CHANNELS = 10
MAP_OBS_N = GRID_SIZE * GRID_SIZE * MAP_CHANNELS

# 8 agents * (x,y,hp) = 24 + 2 base HP = 26 -> pad to 32
PLAYER_OBS_N = 32


class DumobaEnv(gym.Env):
    """
    Dumoba 3v5 environment.

    - Blue controls 3 heroes (RL).
    - Red has 5 scripted heroes.
    - Difficulty modes:
        - easy:   red HP/dmg ~= blue (3v5 but same stats)
        - medium: red HP/dmg buffed
        - hard:   red HP/dmg buffed + red heroes can respawn once
    - Map types:
        - classic: bases at top/bottom center
        - corner:  bases in opposite corners
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        difficulty: str = "easy",
        map_type: str = "classic",
        max_steps: int = 200,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.difficulty = difficulty.lower()
        self.map_type = map_type.lower()
        self.render_mode = render_mode
        self.grid_size = GRID_SIZE
        self.max_steps = max_steps

        # Agent indexing
        self.blue_ids = np.array([0, 1, 2], dtype=np.int32)
        self.red_ids = np.array([3, 4, 5, 6, 7], dtype=np.int32)
        self.n_agents = N_AGENTS

        # Base positions (temporary; finalized in _init_map_layout)
        self.blue_base_pos = np.array([self.grid_size // 2, self.grid_size - 1], dtype=np.int32)
        self.red_base_pos = np.array([self.grid_size // 2, 0], dtype=np.int32)

        # HP and damage base config
        self.max_blue_hero_hp = 10
        self.max_red_hero_hp = 10  # overridden by difficulty
        self.max_base_hp = 20

        self.blue_attack_damage = 2
        self.red_attack_damage = 2  # overridden by difficulty

        # Respawn related
        self.red_respawn_charges = 0
        self.red_respawns_left = np.zeros(len(self.red_ids), dtype=np.int32)
        self.red_spawn_positions = [np.array([0, 0], dtype=np.int32) for _ in self.red_ids]

        # Apply difficulty settings to HP/damage/respawns
        self._set_difficulty_params()

        # Finalize map layout (base positions; hero spawn style is in reset)
        self._init_map_layout()

        # Observation space: flattened map + feature vector
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(MAP_OBS_N + PLAYER_OBS_N,),
            dtype=np.uint8,
        )

        # Action: 3 heroes * MultiDiscrete([7,7,3,2,2,2]) -> 18 dims
        single_hero_nvec = np.array([7, 7, 3, 2, 2, 2], dtype=np.int64)
        self.nvec_single = single_hero_nvec
        self.nvec = np.tile(single_hero_nvec, N_BLUE)
        self.action_space = spaces.MultiDiscrete(self.nvec)

        # Internal state
        self.agent_positions = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.agent_hp = np.zeros((self.n_agents,), dtype=np.int32)
        self.blue_base_hp = None
        self.red_base_hp = None
        self.steps = 0

    # ==================== Config helpers ====================

    def _set_difficulty_params(self):
        """Set HP, damage, respawns based on difficulty mode."""
        if self.difficulty == "easy":
            # 3v5 but symmetric stats
            self.max_blue_hero_hp = 10
            self.max_red_hero_hp = 10
            self.blue_attack_damage = 2
            self.red_attack_damage = 2
            self.red_respawn_charges = 0
        elif self.difficulty == "medium":
            # buffed red
            self.max_blue_hero_hp = 10
            self.max_red_hero_hp = 12
            self.blue_attack_damage = 2
            self.red_attack_damage = 3
            self.red_respawn_charges = 0
        elif self.difficulty == "hard":
            # buffed red + respawns
            self.max_blue_hero_hp = 10
            self.max_red_hero_hp = 12
            self.blue_attack_damage = 2
            self.red_attack_damage = 3
            self.red_respawn_charges = 1
        else:
            raise ValueError(f"Unknown difficulty mode: {self.difficulty}")

    def _init_map_layout(self):
        """Set base positions based on map type."""
        if self.map_type == "classic":
            self.blue_base_pos = np.array(
                [self.grid_size // 2, self.grid_size - 1], dtype=np.int32
            )
            self.red_base_pos = np.array(
                [self.grid_size // 2, 0], dtype=np.int32
            )
        elif self.map_type == "corner":
            # Opposite corners
            self.blue_base_pos = np.array([1, self.grid_size - 2], dtype=np.int32)
            self.red_base_pos = np.array([self.grid_size - 2, 1], dtype=np.int32)
        else:
            raise ValueError(f"Unknown map type: {self.map_type}")

    # ==================== Gym API ====================

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # Reset respawns per difficulty
        self.red_respawns_left[:] = self.red_respawn_charges

        # Spawn blue heroes depending on map type
        if self.map_type == "classic":
            self._spawn_blue_classic()
            self._spawn_red_classic()
        elif self.map_type == "corner":
            self._spawn_blue_corner()
            self._spawn_red_corner()
        else:
            # should not happen due to validation
            self._spawn_blue_classic()
            self._spawn_red_classic()

        # HP
        self.agent_hp[:] = 0
        for bid in self.blue_ids:
            self.agent_hp[bid] = self.max_blue_hero_hp
        for rid in self.red_ids:
            self.agent_hp[rid] = self.max_red_hero_hp

        self.blue_base_hp = self.max_base_hp
        self.red_base_hp = self.max_base_hp
        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (N_BLUE * 6,):
            raise ValueError(f"Expected action shape {(N_BLUE*6,)}, got {action.shape}")

        hero_actions = action.reshape(N_BLUE, 6)

        # 1) Blue movement
        for i, hero_id in enumerate(self.blue_ids):
            if self._is_alive(hero_id):
                self._apply_movement(hero_id, int(hero_actions[i][0]))

        # 2) Red movement (scripted)
        red_actions = []
        for red_id in self.red_ids:
            if self._is_alive(red_id):
                red_action = self._scripted_policy(red_id)
            else:
                red_action = np.zeros(6, dtype=np.int64)
            red_actions.append(red_action)

        red_actions = np.asarray(red_actions, dtype=np.int64)

        for i, red_id in enumerate(self.red_ids):
            if self._is_alive(red_id):
                self._apply_movement(red_id, int(red_actions[i][0]))

        # 3) Attacks
        for i, hero_id in enumerate(self.blue_ids):
            if self._is_alive(hero_id):
                self._apply_attack(hero_id, hero_actions[i])

        for i, red_id in enumerate(self.red_ids):
            if self._is_alive(red_id):
                self._apply_attack(red_id, red_actions[i])

        # Handle red respawns AFTER damage is resolved (only for hard)
        if self.red_respawn_charges > 0:
            self._handle_red_respawns()

        # 4) Done & reward
        terminated, winner = self._check_done()
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
                label = f"{i}"       # 0,1,2
            else:
                label = f"r{i-3}"    # r0..r4
            grid[y][x] = label

        print(f"Step {self.steps}")
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

    # ==================== Spawn helpers ====================

    def _spawn_blue_classic(self):
        cx = self.blue_base_pos[0]
        top_row = self.blue_base_pos[1]
        blue_row = max(0, top_row - 1)
        blue_xs = [cx - 1, cx, cx + 1]
        blue_xs = [int(np.clip(x, 0, self.grid_size - 1)) for x in blue_xs]

        for i, x in enumerate(blue_xs):
            self.agent_positions[self.blue_ids[i]] = np.array([x, blue_row], dtype=np.int32)

    def _spawn_red_classic(self):
        cx_r = self.red_base_pos[0]
        bottom_row = self.red_base_pos[1]
        row1 = min(self.grid_size - 1, bottom_row + 1)
        row2 = min(self.grid_size - 1, bottom_row + 2)
        red_positions = [
            (cx_r - 1, row1),
            (cx_r,     row1),
            (cx_r + 1, row1),
            (cx_r - 1, row2),
            (cx_r + 1, row2),
        ]
        red_positions = [
            (int(np.clip(x, 0, self.grid_size - 1)), int(np.clip(y, 0, self.grid_size - 1)))
            for (x, y) in red_positions
        ]

        self.red_spawn_positions = [np.array(p, dtype=np.int32) for p in red_positions]

        for i, (x, y) in enumerate(red_positions):
            self.agent_positions[self.red_ids[i]] = np.array([x, y], dtype=np.int32)

    def _spawn_blue_corner(self):
        # Blue heroes cluster near blue base but slightly spread
        bx, by = self.blue_base_pos
        positions = [
            (bx, by - 1),
            (bx + 1, by - 1),
            (bx + 1, by),
        ]
        positions = [
            (int(np.clip(x, 0, self.grid_size - 1)), int(np.clip(y, 0, self.grid_size - 1)))
            for (x, y) in positions
        ]
        for i, (x, y) in enumerate(positions):
            self.agent_positions[self.blue_ids[i]] = np.array([x, y], dtype=np.int32)

    def _spawn_red_corner(self):
        rx, ry = self.red_base_pos
        positions = [
            (rx, ry + 1),
            (rx - 1, ry + 1),
            (rx - 1, ry),
            (rx - 2, ry + 1),
            (rx - 2, ry),
        ]
        positions = [
            (int(np.clip(x, 0, self.grid_size - 1)), int(np.clip(y, 0, self.grid_size - 1)))
            for (x, y) in positions
        ]
        self.red_spawn_positions = [np.array(p, dtype=np.int32) for p in positions]

        for i, (x, y) in enumerate(positions):
            self.agent_positions[self.red_ids[i]] = np.array([x, y], dtype=np.int32)

    # ==================== Core helpers ====================

    def _is_alive(self, agent_id: int) -> bool:
        return self.agent_hp[agent_id] > 0

    def _in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _move(self, pos, direction: int):
        x, y = pos
        # 0 = stay; 1 = up; 2 = down; 3 = left; 4 = right
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

    def _attack_one_target(self, attacker_id: int, target_pos, target_hp: int, dmg: int):
        if target_hp <= 0:
            return target_hp
        if self._manhattan(self.agent_positions[attacker_id], target_pos) <= 1:
            return max(0, target_hp - dmg)
        return target_hp

    def _apply_attack(self, agent_id: int, action_vec: np.ndarray):
        attack_flag = int(action_vec[3])
        if attack_flag != 1:
            return

        if agent_id in self.blue_ids:
            enemy_ids = self.red_ids
            enemy_base_pos = self.red_base_pos
            base_hp_attr = "red_base_hp"
            dmg = self.blue_attack_damage
        else:
            enemy_ids = self.blue_ids
            enemy_base_pos = self.blue_base_pos
            base_hp_attr = "blue_base_hp"
            dmg = self.red_attack_damage

        # Attack enemy heroes
        for eid in enemy_ids:
            if self.agent_hp[eid] > 0:
                self.agent_hp[eid] = self._attack_one_target(
                    agent_id, self.agent_positions[eid], self.agent_hp[eid], dmg
                )

        # Attack enemy base
        base_hp = getattr(self, base_hp_attr)
        base_hp = self._attack_one_target(agent_id, enemy_base_pos, base_hp, dmg)
        setattr(self, base_hp_attr, base_hp)

    def _handle_red_respawns(self):
        """Respawn dead red heroes if they still have respawns left."""
        for idx, rid in enumerate(self.red_ids):
            if self.agent_hp[rid] <= 0 and self.red_respawns_left[idx] > 0:
                # Consume one respawn
                self.red_respawns_left[idx] -= 1
                # Respawn at original spawn position with full red HP
                self.agent_positions[rid] = self.red_spawn_positions[idx].copy()
                self.agent_hp[rid] = self.max_red_hero_hp

    def _nearest_enemy(self, agent_id: int):
        # For blue: chase red. For red: chase blue.
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

        # If no enemy heroes alive, chase enemy base
        if best_pos is None:
            best_pos = self.red_base_pos if agent_id in self.blue_ids else self.blue_base_pos

        return best_pos

    def _scripted_policy(self, agent_id: int) -> np.ndarray:
        """Scripted red/blue policy (used only for red in training)."""
        action = np.zeros(6, dtype=np.int64)
        if not self._is_alive(agent_id):
            return action

        target_pos = self._nearest_enemy(agent_id)
        my_pos = self.agent_positions[agent_id]

        dx = np.sign(target_pos[0] - my_pos[0])
        dy = np.sign(target_pos[1] - my_pos[1])

        if dy > 0:
            move_dir = 1
        elif dy < 0:
            move_dir = 2
        elif dx < 0:
            move_dir = 3
        elif dx > 0:
            move_dir = 4
        else:
            move_dir = 0

        action[0] = move_dir

        # Attack flag if in range of any enemy or enemy base
        if agent_id in self.blue_ids:
            enemy_ids = self.red_ids
            base_pos = self.red_base_pos
        else:
            enemy_ids = self.blue_ids
            base_pos = self.blue_base_pos

        in_range = False
        for eid in enemy_ids:
            if self.agent_hp[eid] > 0 and self._manhattan(my_pos, self.agent_positions[eid]) <= 1:
                in_range = True
                break
        if not in_range and self._manhattan(my_pos, base_pos) <= 1:
            in_range = True

        action[3] = 1 if in_range else 0
        return action

    def _check_done(self):
        # Blue: still in game if any alive
        blue_alive = any(self.agent_hp[i] > 0 for i in self.blue_ids)

        # Red: in game if alive OR has respawns left (for hard)
        red_still_in_game = False
        for idx, rid in enumerate(self.red_ids):
            if self.agent_hp[rid] > 0:
                red_still_in_game = True
                break
            if self.red_respawns_left[idx] > 0:
                red_still_in_game = True
                break

        # Blue loses if base dies or all heroes dead
        if (self.blue_base_hp <= 0) or (not blue_alive):
            return True, "red"

        # Blue wins if red base dies OR no red heroes + no respawns left
        if (self.red_base_hp <= 0) or (not red_still_in_game):
            return True, "blue"

        return False, None

    def _compute_reward(self, terminated: bool, winner):
        if not terminated:
            return -0.01
        if winner == "blue":
            return 1.0
        elif winner == "red":
            return -1.0
        return 0.0

    def _get_obs(self) -> np.ndarray:
        """
        Build observation:
        - Flattened one-hot map: (GRID_SIZE, GRID_SIZE, MAP_CHANNELS)
        - Feature vector: (PLAYER_OBS_N,)
        """
        # Map grid: one-hot occupancy
        map_grid = np.zeros((self.grid_size, self.grid_size, MAP_CHANNELS), dtype=np.uint8)

        # Blue heroes channels 0–2
        for idx, agent_id in enumerate(self.blue_ids):
            if self.agent_hp[agent_id] > 0:
                x, y = self.agent_positions[agent_id]
                map_grid[y, x, idx] = 1

        # Red heroes channels 3–7
        for idx, agent_id in enumerate(self.red_ids):
            if self.agent_hp[agent_id] > 0:
                x, y = self.agent_positions[agent_id]
                map_grid[y, x, 3 + idx] = 1

        # Bases
        if self.blue_base_hp > 0:
            bx, by = self.blue_base_pos
            map_grid[by, bx, 8] = 1
        if self.red_base_hp > 0:
            rx, ry = self.red_base_pos
            map_grid[ry, rx, 9] = 1

        map_flat = map_grid.reshape(-1)  # length MAP_OBS_N

        # Features: (x,y,hp) for each agent + base HPs
        feats = []
        for agent_id in range(self.n_agents):
            feats.append(int(self.agent_positions[agent_id][0]))
            feats.append(int(self.agent_positions[agent_id][1]))
            feats.append(int(self.agent_hp[agent_id]))
        feats.append(int(self.blue_base_hp))
        feats.append(int(self.red_base_hp))

        # Pad/truncate to PLAYER_OBS_N
        if len(feats) < PLAYER_OBS_N:
            feats.extend([0] * (PLAYER_OBS_N - len(feats)))
        elif len(feats) > PLAYER_OBS_N:
            feats = feats[:PLAYER_OBS_N]

        feats_arr = np.array(feats, dtype=np.uint8)

        obs = np.concatenate([map_flat, feats_arr], axis=0).astype(np.uint8)
        return obs


if __name__ == "__main__":
    # Quick manual test
    env = DumobaEnv(difficulty="hard", map_type="classic", render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        a = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(a)
        if done or truncated:
            print("Episode finished. Reward:", reward, "info:", info)
            break
    env.close()
