import random
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    class _Discrete:
        def __init__(self, n): self.n = n
        def sample(self): return random.randint(0, self.n - 1)
    class _Box:
        def __init__(self, **kw): pass
    class _spaces:
        Discrete = _Discrete; Box = _Box
    class _gym:
        Env = object; spaces = _spaces()
    gym = _gym(); spaces = _spaces()




PRODUCT_CATALOGUE: Dict[str, List[Dict]] = {
    "> Dairy":     [{"name": "Whole Milk",      "aisle": "A1", "reward": 10},
                   {"name": "Cheddar Cheese",   "aisle": "A1", "reward": 15},
                   {"name": "Curd",             "aisle": "A1", "reward": 12}],
    "> Bakery":    [{"name": "Bread",           "aisle": "A2", "reward": 10},
                   {"name": "Cake",             "aisle": "A2", "reward":  8},
                   {"name": "Pastry",           "aisle": "A2", "reward":  7}],
    "> Fruits":    [{"name": "Apples",          "aisle": "A3", "reward": 10},
                   {"name": "Strawberry",       "aisle": "A3", "reward": 12},
                   {"name": "Kiwi",             "aisle": "A3", "reward": 15}],
    "> Frozen":    [{"name": "Frozen Pizza",    "aisle": "A4", "reward":  8},
                   {"name": "Ice Cream Tub",    "aisle": "A4", "reward": 10},
                   {"name": "Veggie Burger",    "aisle": "A4", "reward": 11}],
    "> Beverages": [{"name": "Orange Juice",    "aisle": "A5", "reward":  9},
                   {"name": "Bisleri",          "aisle": "A5", "reward":  6},
                   {"name": "Cold Coffee",      "aisle": "A5", "reward": 14}],
    "> Snacks":    [{"name": "Lays Chips",      "aisle": "A6", "reward":  7},
                   {"name": "Mix Namkeen",      "aisle": "A6", "reward":  9},
                   {"name": "Dark Chocolate",   "aisle": "A6", "reward": 13}],
}

ALL_PRODUCTS: Dict[str, Dict] = {}
for _cat, _items in PRODUCT_CATALOGUE.items():
    for _item in _items:
        ALL_PRODUCTS[_item["name"].lower()] = {**_item, "category": _cat}

AISLE_GRID_POS: Dict[str, Tuple[int, int]] = {
    "A1": (1, 1), "A2": (3, 1), "A3": (5, 1),
    "A4": (1, 5), "A5": (3, 5), "A6": (5, 5),
}
AISLE_SYMBOL: Dict[str, str] = {
    "A1": "D", "A2": "B", "A3": "P",
    "A4": "F", "A5": "V", "A6": "S",
}

LEVEL_CONFIG: Dict[str, Dict] = {
    "easy": {
        "label":            "EASY",
        "description":      "1 product · 200 steps · 2× rewards · walk to checkout",
        "num_products":     1,
        "max_steps":        200,
        "reward_mult":      2.0,
        "step_penalty":     -0.5,
        "wrong_penalty":    0.0,
        "shaping_scale":    1.0,
        "completion_bonus": 20.0,
        "checkout_bonus":   15.0,
        "closest_rule":     False,
    },
    "medium": {
        "label":            "MEDIUM",
        "description":      "3 products · 80 steps · any order · walk to checkout",
        "num_products":     3,
        "max_steps":        80,
        "reward_mult":      1.0,
        "step_penalty":     -1.0,
        "wrong_penalty":    -5.0,
        "shaping_scale":    1.5,
        "completion_bonus": 35.0,
        "checkout_bonus":   20.0,
        "closest_rule":     False,
    },
    "hard": {
        "label":            "HARD",
        "description":      "4 products · 50 steps · CLOSEST first · walk to checkout",
        "num_products":     4,
        "max_steps":        50,
        "reward_mult":      1.5,
        "step_penalty":     -2.0,
        "wrong_penalty":    -15.0,
        "shaping_scale":    2.0,
        "completion_bonus": 50.0,
        "checkout_bonus":   30.0,
        "closest_rule":     True,
    },
}

ACTION_DELTAS  = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES   = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]
COLLECT_ACTION = 4
N_ACTIONS      = 5
MAX_PRODUCTS   = 4
OBS_LEN        = 4 + 3 * MAX_PRODUCTS  # 16



def _max_possible_reward(cfg: Dict, products: List[Dict]) -> float:
    """
    Theoretical upper bound for a perfect episode (no step penalties,
    all collect bonuses + completion_bonus + checkout_bonus).
    Used to normalise the final score to [0, 1].
    """
    item_rewards = sum(p["reward"] * cfg["reward_mult"] for p in products)
    return item_rewards + cfg["completion_bonus"] + cfg["checkout_bonus"]



class SupermarketEnv:
    GRID_ROWS = 8
    GRID_COLS = 7
    SPAWN_POS = (0, 3)
    EXIT_POS  = (7, 3)

    def __init__(self, level: str = "easy",
                 products: Optional[List[str]] = None,
                 render_mode: Optional[str] = None):
        if level not in LEVEL_CONFIG:
            raise ValueError(f"level must be one of {list(LEVEL_CONFIG)}")
        self.level           = level
        self.cfg             = LEVEL_CONFIG[level]
        self._fixed_products = products
        self.render_mode     = render_mode

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_LEN,), dtype=np.float32)
            self.action_space = spaces.Discrete(N_ACTIONS)
        else:
            self.action_space = _Discrete(N_ACTIONS)

        self._agent_pos:       Tuple[int, int] = self.SPAWN_POS
        self._target_products: List[Dict]      = []
        self._collected:       List[bool]      = []
        self._phase:           int             = 0
        self._steps_taken:     int             = 0
        self._closest_idx:     int             = 0
        self._prev_dist:       int             = 0
        self._total_reward:    float           = 0.0
        self._reward_log:      List[Dict]      = []
        self._task_status:     str             = "In-Progress"
        self._max_reward:      float           = 1.0   # set properly after reset

    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        if options:
            if "level" in options:
                self.level = options["level"]
                self.cfg   = LEVEL_CONFIG[self.level]
            if "products" in options:
                self._fixed_products = options["products"]

        self._target_products = self._resolve_products()
        self._agent_pos    = self.SPAWN_POS
        self._collected    = [False] * len(self._target_products)
        self._phase        = 0
        self._steps_taken  = 0
        self._total_reward = 0.0
        self._reward_log   = []
        self._task_status  = "In-Progress"
        self._closest_idx  = self._compute_closest_idx()
        self._prev_dist    = self._dist_to_goal()
        self._max_reward   = _max_possible_reward(self.cfg, self._target_products)

        return self._build_obs(), self._build_info("reset", 0.0)

    def step(self, action: int):
        reward     = 0.0
        terminated = False
        truncated  = False
        event      = ""

        reward            += self.cfg["step_penalty"]
        self._steps_taken += 1

        if action in ACTION_DELTAS:
            dr, dc = ACTION_DELTAS[action]
            nr     = self._agent_pos[0] + dr
            nc     = self._agent_pos[1] + dc

            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                self._agent_pos   = (nr, nc)
                new_dist          = self._dist_to_goal()
                reward           += self.cfg["shaping_scale"] * (self._prev_dist - new_dist)
                self._prev_dist   = new_dist
                self._closest_idx = self._compute_closest_idx()
                event = f"MOVE_{ACTION_NAMES[action]} → ({nr},{nc})"

                if self._phase == 1 and self._agent_pos == self.EXIT_POS:
                    reward           += self.cfg["checkout_bonus"]
                    self._task_status = "Success"
                    terminated        = True
                    event            += f" | REACHED COUNTER ✅ +{self.cfg['checkout_bonus']}"
            else:
                reward -= 1.0
                event   = f"WALL_HIT (tried {ACTION_NAMES[action]})"

        elif action == COLLECT_ACTION:
            if self._phase == 0:
                r, event  = self._try_collect()
                reward   += r
                self._closest_idx = self._compute_closest_idx()

                if all(self._collected):
                    reward         += self.cfg["completion_bonus"]
                    self._phase     = 1
                    self._prev_dist = self._dist_to_goal()
                    event          += (f" | ALL COLLECTED ✅ +{self.cfg['completion_bonus']}"
                                       f" → RETURN TO COUNTER at {self.EXIT_POS}")

                    if self._agent_pos == self.EXIT_POS:
                        reward           += self.cfg["checkout_bonus"]
                        self._task_status = "Success"
                        terminated        = True
                        event            += f" | INSTANT CHECKOUT ✅ +{self.cfg['checkout_bonus']}"
                else:
                    self._prev_dist = self._dist_to_goal()
            else:
                event = "COLLECT ignored — already in exit phase, head to counter"

        if not terminated and self._steps_taken >= self.cfg["max_steps"]:
            reward           -= 20.0
            self._task_status = "Failed"
            truncated         = True
            event            += " | BUDGET_EXCEEDED −20"

        self._total_reward += reward
        self._append_log(action, reward, event)
        info = self._build_info(event, reward)
        return self._build_obs(), float(reward), terminated, truncated, info

   def normalised_score(self) -> float:
        """Calculate normalised score, ensuring it's strictly between 0 and 1.
        
        Returns a score in [0.001, 0.999] to satisfy OpenEnv evaluation constraints.
        This prevents edge cases where perfect/failed episodes return exactly 0.0 or 1.0.
        """
        raw = max(0.0, self._total_reward)
        
        # Calculate base score with proper rounding
        if self._max_reward > 0:
            score = round(min(raw / self._max_reward, 1.0), 6)
        else:
            score = 0.0
        
        # Clamp strictly to (0, 1) range - never return exact 0.0 or 1.0
        epsilon = 0.001
        clamped = max(epsilon, min(1.0 - epsilon, score))
        
        return float(clamped)
    
    def action_masks(self) -> np.ndarray:
        mask     = np.ones(N_ACTIONS, dtype=bool)
        on_aisle = self._agent_pos in AISLE_GRID_POS.values()
        if not on_aisle or self._phase == 1:
            mask[COLLECT_ACTION] = False
        else:

            aisle = next(
                (code for code, pos in AISLE_GRID_POS.items() if self._agent_pos == pos),
                None,
            )

            pending = [
                i for i, (p, c) in enumerate(zip(self._target_products, self._collected))
                if not c and p["aisle"] == aisle
            ]
            if not pending:
                mask[COLLECT_ACTION] = False
            elif self.cfg["closest_rule"]:
                # On HARD level: only allow COLLECT at the closest aisle
                req_aisle = self._target_products[self._closest_idx]["aisle"]
                if aisle != req_aisle:
                    mask[COLLECT_ACTION] = False
        return mask

    def get_state_key(self) -> tuple:
        return (
            self._agent_pos[0],
            self._agent_pos[1],
            self._phase,
            tuple(self._collected),
            self._closest_idx,
        )

    def render(self):
        if self.render_mode != "ansi":
            return None
        grid = [["  · "] * self.GRID_COLS for _ in range(self.GRID_ROWS)]
        for code, (r, c) in AISLE_GRID_POS.items():
            sym = AISLE_SYMBOL[code]
            is_pending = any(not self._collected[i]
                             and self._target_products[i]["aisle"] == code
                             for i in range(len(self._target_products)))
            is_closest = (not all(self._collected)
                          and self._target_products[self._closest_idx]["aisle"] == code
                          and self._phase == 0)
            if is_closest:
                grid[r][c] = " [★] "
            elif is_pending:
                grid[r][c] = f" [{sym}] "
            else:
                grid[r][c] = f"  {sym.lower()}  "

        er, ec = self.EXIT_POS
        grid[er][ec] = "[CTR]" if self._phase == 1 else " ctr "

        ar, ac = self._agent_pos
        grid[ar][ac] = "  @  "

        phase_str = ("Phase 0: COLLECTING PRODUCTS" if self._phase == 0
                     else "Phase 1: RETURN TO COUNTER")
        print()
        print(f"  ╔═══ {self.level.upper()} ═══ Step {self._steps_taken}/{self.cfg['max_steps']} "
              f"═══ Reward: {self._total_reward:.1f} ═══ {self._task_status} ╗")
        print(f"  ║  {phase_str:<55}║")
        print(f"  ╠{'═'*62}╣")
        header = "   " + "".join(f"  {c}   " for c in range(self.GRID_COLS))
        print(f"  ║ {header}║")
        print(f"  ╠{'═'*62}╣")
        for r, row in enumerate(grid):
            print(f"  ║ {r} {''.join(row)} ║")
        print(f"  ╚{'═'*62}╝")

    def close(self): pass

    @property
    def task_status(self):     return self._task_status
    @property
    def reward_log(self):      return self._reward_log
    @property
    def total_reward(self):    return self._total_reward
    @property
    def target_names(self):    return [p["name"] for p in self._target_products]
    @property
    def collected_names(self): return [p["name"] for p, c in
                                       zip(self._target_products, self._collected) if c]


    def _manhattan(self, cell: Tuple[int, int]) -> int:
        return abs(self._agent_pos[0] - cell[0]) + abs(self._agent_pos[1] - cell[1])

    def _dist_to_goal(self) -> int:
        if self._phase == 1 or all(self._collected):
            return self._manhattan(self.EXIT_POS)
        return self._manhattan(AISLE_GRID_POS[self._target_products[self._closest_idx]["aisle"]])

    def _compute_closest_idx(self) -> int:
        best_i, best_d = 0, float("inf")
        for i, (p, c) in enumerate(zip(self._target_products, self._collected)):
            if c:
                continue
            d = self._manhattan(AISLE_GRID_POS[p["aisle"]])
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _try_collect(self) -> Tuple[float, str]:
        aisle = next(
            (code for code, pos in AISLE_GRID_POS.items() if self._agent_pos == pos),
            None,
        )
        if aisle is None:
            return self.cfg["wrong_penalty"], "COLLECT_FAIL: not standing on an aisle"

        candidates = [
            i for i, (p, c) in enumerate(zip(self._target_products, self._collected))
            if not c and p["aisle"] == aisle
        ]
        if not candidates:
            return self.cfg["wrong_penalty"], f"COLLECT_FAIL: no pending target at {aisle}"

        if self.cfg["closest_rule"]:
            req_aisle = self._target_products[self._closest_idx]["aisle"]
            if aisle != req_aisle:
                d_here = self._manhattan(AISLE_GRID_POS[aisle])
                d_req  = self._manhattan(AISLE_GRID_POS[req_aisle])
                return (
                    self.cfg["wrong_penalty"],
                    f"CLOSEST_RULE_FAIL: went to {aisle}({d_here} steps) "
                    f"but {req_aisle}({d_req} steps) is closest",
                )

        idx = candidates[0]
        self._collected[idx] = True
        item_reward = self._target_products[idx]["reward"] * self.cfg["reward_mult"]
        return item_reward, f"COLLECTED {self._target_products[idx]['name']} +{item_reward:.1f}"

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_LEN, dtype=np.float32)
        obs[0] = self._agent_pos[0] / (self.GRID_ROWS - 1)
        obs[1] = self._agent_pos[1] / (self.GRID_COLS - 1)
        obs[2] = float(self._phase)
        obs[3] = self._steps_taken / self.cfg["max_steps"]
        for i in range(MAX_PRODUCTS):
            base = 4 + i * 3
            if i < len(self._target_products):
                p = self._target_products[i]
                gr, gc = AISLE_GRID_POS[p["aisle"]]
                obs[base]     = gr / (self.GRID_ROWS - 1)
                obs[base + 1] = gc / (self.GRID_COLS - 1)
                obs[base + 2] = float(self._collected[i])
        return obs

    def _build_info(self, event: str, step_reward: float) -> Dict[str, Any]:
        return {
            "task_status":      self._task_status,
            "reward_log":       self._reward_log,
            "total_reward":     round(self._total_reward, 4),
            "normalised_score": self.normalised_score(),
            "step_reward":      round(step_reward, 4),
            "steps_taken":      self._steps_taken,
            "steps_remaining":  self.cfg["max_steps"] - self._steps_taken,
            "inventory":        self.collected_names,
            "targets":          self.target_names,
            "level":            self.level,
            "phase":            self._phase,
            "phase_label":      "collecting" if self._phase == 0 else "returning_to_counter",
            "event":            event,
            "action_mask":      self.action_masks().tolist(),
            "agent_pos":        list(self._agent_pos),
            "closest_target":   (self._target_products[self._closest_idx]["name"]
                                 if not all(self._collected) else None),
            "closest_rule":     self.cfg["closest_rule"],
            "obs_len":          OBS_LEN,
        }

    def _append_log(self, action: int, reward: float, event: str):
        self._reward_log.append({
            "step":         self._steps_taken,
            "action":       ACTION_NAMES[action],
            "step_reward":  round(reward, 4),
            "total_reward": round(self._total_reward, 4),
            "position":     list(self._agent_pos),
            "phase":        self._phase,
            "inventory":    self.collected_names.copy(),
            "task_status":  self._task_status,
            "event":        event,
        })

    def _resolve_products(self) -> List[Dict]:
        n = self.cfg["num_products"]
        if self._fixed_products:
            out = []
            for name in self._fixed_products[:n]:
                key = name.lower()
                if key not in ALL_PRODUCTS:
                    raise ValueError(f"Unknown product '{name}'")
                out.append(ALL_PRODUCTS[key])
            return out
        return random.sample(list(ALL_PRODUCTS.values()), min(n, len(ALL_PRODUCTS)))



if GYM_AVAILABLE:
    for _lvl in ("easy", "medium", "hard"):
        try:
            gym.register(
                f"SupermarketNav-{_lvl.capitalize()}-v0",
                entry_point="supermart_env:SupermarketEnv",
                kwargs={"level": _lvl},
            )
        except Exception:
            pass
