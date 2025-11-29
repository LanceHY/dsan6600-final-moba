import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

MAP_OBS_N = 11 * 11 * 4   # 11x11 map, 4 channels
PLAYER_OBS_N = 26         # player/features vector


class DumobaEnv(gym.Env):
    """
    Dumoba: A simplified MOBA-like environment inspired by the PufferLib MOBA API.

    Observation:
      - Flattened 11x11x4 map (uint8)
          ch0: hero position
          ch1: enemy position
          ch2: hero base
          ch3: enemy base
      - 26-dim player feature vector (hp, base hp, positions, etc., padded)

    Action (MultiDiscrete [7,7,3,2,2,2]):
      a[0]: movement direction
            0 = stay
            1 = up
            2 = down
            3 = left
            4 = right
            5,6 = treated as stay for now
      a[1]: reserved (ignored in v0, can be used later)
      a[2]: reserved for target selection (ignored in v0)
      a[3]: attack flag (0 = no attack, 1 = attack if in range)
      a[4]: reserved ability flag (ignored in v0)
      a[5]: reserved ability flag (ignored in v0)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, max_steps: int = 100):
        super().__init__()

        self.grid_size = 11
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Basic stats
        self.max_hero_hp = 10
        self.max_enemy_hp = 10
        self.max_base_hp = 15
        self.attack_damage = 2

        # Fixed base positions (center top and center bottom)
        self.hero_base_pos = np.array([self.grid_size // 2, self.grid_size - 1], dtype=np.int32)
        self.enemy_base_pos = np.array([self.grid_size // 2, 0], dtype=np.int32)

        # Observation space: uint8, like the PufferLib MOBA example
        self.single_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(MAP_OBS_N + PLAYER_OBS_N,),
            dtype=np.uint8,
        )
        self.observation_space = self.single_observation_space

        # Action space: MultiDiscrete with same shape as your snippet
        self.single_action_space = spaces.MultiDiscrete([7, 7, 3, 2, 2, 2])
        self.action_space = self.single_action_space

        # Internal state
        self.hero_pos = None
        self.enemy_pos = None
        self.hero_hp = None
        self.enemy_hp = None
        self.hero_base_hp = None
        self.enemy_base_hp = None
        self.steps = 0

    # ------------ Gym API ------------ #

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # Start heroes on top of their bases
        self.hero_pos = self.hero_base_pos.copy()
        self.enemy_pos = self.enemy_base_pos.copy()

        self.hero_hp = self.max_hero_hp
        self.enemy_hp = self.max_enemy_hp
        self.hero_base_hp = self.max_base_hp
        self.enemy_base_hp = self.max_base_hp

        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        self.steps += 1

        # 1) Apply hero action
        self._apply_hero_action(action)

        # 2) Enemy scripted policy
        enemy_action = self._enemy_policy()
        self._apply_enemy_action(enemy_action)

        # 3) Check terminal conditions
        terminated, winner = self._check_done()

        # 4) Reward from hero's perspective
        reward = self._compute_reward(terminated, winner)

        # 5) Truncation based on max_steps
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
        bx, by = self.hero_base_pos
        grid[by][bx] = "B"
        ex, ey = self.enemy_base_pos
        grid[ey][ex] = "b"

        # Units
        if self.hero_hp > 0:
            hx, hy = self.hero_pos
            grid[hy][hx] = "H"
        if self.enemy_hp > 0:
            ex, ey = self.enemy_pos
            grid[ey][ex] = "E"

        print(f"Step {self.steps}")
        for row in reversed(grid):
            print(" ".join(row))
        print(
            f"Hero HP: {self.hero_hp}, Base HP: {self.hero_base_hp} | "
            f"Enemy HP: {self.enemy_hp}, Base HP: {self.enemy_base_hp}"
        )
        print()

    def close(self):
        pass

    # ------------ Helpers: Obs & Encoding ------------ #

    def _get_obs(self) -> np.ndarray:
        """
        Returns a uint8 vector of length MAP_OBS_N + PLAYER_OBS_N
        """
        # Map encoding: 4 channels (hero, enemy, hero_base, enemy_base)
        map_obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.uint8)

        # hero
        if self.hero_hp > 0:
            x, y = self.hero_pos
            map_obs[y, x, 0] = 1

        # enemy
        if self.enemy_hp > 0:
            x, y = self.enemy_pos
            map_obs[y, x, 1] = 1

        # hero base
        x, y = self.hero_base_pos
        if self.hero_base_hp > 0:
            map_obs[y, x, 2] = 1

        # enemy base
        x, y = self.enemy_base_pos
        if self.enemy_base_hp > 0:
            map_obs[y, x, 3] = 1

        map_flat = map_obs.flatten()  # length = MAP_OBS_N

        # Player feature vector (pad to 26)
        features = [
            self.hero_pos[0],
            self.hero_pos[1],
            self.hero_hp,
            self.enemy_pos[0],
            self.enemy_pos[1],
            self.enemy_hp,
            self.hero_base_hp,
            self.enemy_base_hp,
        ]
        # Pad with zeros to length 26
        if len(features) < PLAYER_OBS_N:
            features.extend([0] * (PLAYER_OBS_N - len(features)))

        features = np.array(features, dtype=np.uint8)

        obs = np.concatenate([map_flat, features], dtype=np.uint8)
        assert obs.shape[0] == MAP_OBS_N + PLAYER_OBS_N
        return obs

    # ------------ Movement & Combat ------------ #

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

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _attack_if_in_range(self, attacker_pos, target_pos, target_hp):
        if self._manhattan(attacker_pos, target_pos) <= 1 and target_hp > 0:
            return max(0, target_hp - self.attack_damage)
        return target_hp

    # ------------ Applying Actions ------------ #

    def _apply_hero_action(self, action: np.ndarray):
        """
        Interpret hero action MultiDiscrete[7,7,3,2,2,2].
        For v0:
          - a[0]: movement direction (0..4)
          - a[3]: attack flag
        """
        move_dir = int(action[0])
        attack_flag = int(action[3])

        # Movement
        if move_dir in [1, 2, 3, 4]:
            self.hero_pos = self._move(self.hero_pos, move_dir)

        # Attack
        if attack_flag == 1:
            # Attack enemy hero
            self.enemy_hp = self._attack_if_in_range(
                self.hero_pos, self.enemy_pos, self.enemy_hp
            )
            # Attack enemy base if in range
            self.enemy_base_hp = self._attack_if_in_range(
                self.hero_pos, self.enemy_base_pos, self.enemy_base_hp
            )

    def _enemy_policy(self) -> np.ndarray:
        """
        Simple scripted enemy policy that approximates the same action space.
        - Move greedily toward hero
        - Attack if in range
        """
        action = np.zeros(6, dtype=np.int64)

        # Movement choice (a[0])
        ex, ey = self.enemy_pos
        hx, hy = self.hero_pos
        dx = np.sign(hx - ex)
        dy = np.sign(hy - ey)

        # Prefer vertical moves first
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

        # Attack flag (a[3]) if in range of hero or hero base
        if (
            self._manhattan(self.enemy_pos, self.hero_pos) <= 1
            or self._manhattan(self.enemy_pos, self.hero_base_pos) <= 1
        ):
            action[3] = 1  # attack
        else:
            action[3] = 0

        # Other components left at 0 (can be extended later)
        return action

    def _apply_enemy_action(self, action: np.ndarray):
        move_dir = int(action[0])
        attack_flag = int(action[3])

        if move_dir in [1, 2, 3, 4]:
            self.enemy_pos = self._move(self.enemy_pos, move_dir)

        if attack_flag == 1:
            self.hero_hp = self._attack_if_in_range(
                self.enemy_pos, self.hero_pos, self.hero_hp
            )
            self.hero_base_hp = self._attack_if_in_range(
                self.enemy_pos, self.hero_base_pos, self.hero_base_hp
            )

    # ------------ Termination & Reward ------------ #

    def _check_done(self):
        """
        Returns (terminated: bool, winner: 'hero' | 'enemy' | None)
        """
        # Hero side dead
        if self.hero_hp <= 0 or self.hero_base_hp <= 0:
            return True, "enemy"

        # Enemy side dead
        if self.enemy_hp <= 0 or self.enemy_base_hp <= 0:
            return True, "hero"

        return False, None

    def _compute_reward(self, terminated: bool, winner):
        if not terminated:
            # Small step cost
            return -0.01

        if winner == "hero":
            return 1.0
        elif winner == "enemy":
            return -1.0

        return 0.0


if __name__ == "__main__":
    # Simple manual test: random agent
    env = DumobaEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            print("Episode finished. Reward:", reward, "Info:", info)
            break
