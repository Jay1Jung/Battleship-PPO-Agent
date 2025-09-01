# battleship_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rules import length_constraint, hit_technique
import battleship_original as bo

C_UNKNOWN, C_HIT, C_MISS = 0, 1, 2

class BattleshipEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, board_size=10, max_steps=100, seed=None):
        super().__init__()
        self.N = board_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=0, high=2, shape=(self.N, self.N), dtype=np.int8)
        self.action_space = spaces.Discrete(self.N * self.N)

        self.obs = None         # (N,N) 0/1/2
        self._opp = None        # (N,N) 'O' or ship letter (A/B/C/D/S)
        self.steps = 0

    def _place_random(self):

        N = self.N

        if hasattr(bo, "opp_address"):
            bo.opp_address.clear()

        
        opp_map_list = [["O"] * N for _ in range(N)]

        
        bo.comp_set_location_aircraft(opp_map_list)    
        bo.comp_set_location_battleship(opp_map_list)  
        bo.comp_set_location_destroyer(opp_map_list)   
        bo.comp_set_location_submarine(opp_map_list)   
        bo.comp_set_location_cruiser(opp_map_list)    

        # convert to numpy array
        return np.array(opp_map_list, dtype="<U1")

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._opp = self._place_random()                              # 'O' / 'A''B''C''D''S'
        self.obs = np.zeros((self.N, self.N), dtype=np.int8)          # 전부 UNKNOWN
        self.steps = 0
        return self.obs.copy(), {}

    def step(self, action: int):
        r, c = divmod(int(action), self.N)
        self.steps += 1
        reward, done = 0.0, False
        info = {}

        # big penalty for already hit/miss
        if self.obs[r, c] != C_UNKNOWN:
            return self.obs.copy(), -1.0, False, False, info

        cell = self._opp[r, c]
        if cell in {"A", "B", "C", "D", "S"}: 
            self.obs[r, c] = C_HIT
            reward = 1.0

            # all same letter = sink 
            sunk = np.all(self.obs[self._opp == cell] == C_HIT)
            if sunk:
                reward += 5.0

            # all sink = win
            all_ship_mask = np.isin(self._opp, list("ABCDS"))
            if np.all(self.obs[all_ship_mask] == C_HIT):
                reward += 10.0
                done = True
        else:
            # MISS
            self.obs[r, c] = C_MISS
            reward = -0.05

        if self.steps >= self.max_steps:
            done = True

        return self.obs.copy(), reward, done, False, info

    # ---------------- valid action mask ----------------
    def action_mask(self, use_length_rule=True, use_hit_focus=True):
        unknown = (self.obs.reshape(-1) == C_UNKNOWN)
        base = unknown
        
        masks_by_len = None
        if use_length_rule:
            masks_by_len = length_constraint(self.obs) 
            if masks_by_len:
                Lmax = max(masks_by_len.keys())
                base &= masks_by_len[Lmax].reshape(-1)

        if use_hit_focus and (self.obs == C_HIT).any():
            hitm = hit_technique(self.obs)
            if hitm.any():
                focus = base & hitm
                if focus.any():
                    return focus.astype(bool)
                

        base2 = unknown
        if use_length_rule and masks_by_len:
            Lmax = max(masks_by_len.keys())
            base2 &= masks_by_len[Lmax].reshape(-1)
        if not base2.any():
            base2 = unknown

        return base2.astype(bool) 