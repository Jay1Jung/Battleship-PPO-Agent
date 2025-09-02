# battleship_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import battleship_original as bo
import importlib
bo = importlib.reload(bo) 

if not hasattr(bo, "opp_address"):
    bo.opp_address = []
opp_address = bo.opp_address



C_UNKNOWN, C_HIT, C_MISS = 0, 1, 2

SHIP_LENGTH = {"A" : 5, "B" : 4, "S" : 3, "D" : 3, "C" :2}

class BattleshipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, board_size=10, max_steps=100, seed=None):
        super().__init__()
        self.N = board_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=0, high=2, shape=(self.N, self.N), dtype=np.int8)
        self.action_space = spaces.Discrete(self.N * self.N)

        self.obs = None
        self._opp = None
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
        return np.array(opp_map_list, dtype="<U1")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._opp = self._place_random()
        self.obs = np.zeros((self.N, self.N), dtype=np.int8)
        self.steps = 0
        return self.obs.copy(), {}

    def step(self, action: int):
        
        mask = self.action_mask()
        if not mask.any():
            
            return self.obs.copy(), 0.0, True, False, {"no_actions": True}
        
        if not (0 <= action < self.N * self.N):
            return self.obs.copy(), -1.0, False, False, {"illegal": True}

        if not mask[action]:
            return self.obs.copy(), -1.0, False, False, {"illegal": True}

        r, c = divmod(int(action), self.N)
        self.steps += 1

        reward = 0.0

        event = None

        step_penalty = -1.0 
        cell = self._opp[r, c]
        if cell in {"A", "B", "C", "D", "S"}:
            self.obs[r, c] = C_HIT


            sunk = np.all(self.obs[(self._opp == cell)] == C_HIT)
            if sunk:
                event = ("SINK" , SHIP_LENGTH[cell])
                reward = step_penalty + 12.0
            else:
                event = "HIT"
                reward = step_penalty + 4.0
        else:
            self.obs[r, c] = C_MISS
            event = "MISS"
            reward = step_penalty

        all_ship_mask = np.isin(self._opp, ["A", "B", "D","S","C"])
        terminated = bool(np.all(self.obs[all_ship_mask] == C_HIT))   # 모든 함선 격침
        truncated  = (self.steps >= self.max_steps)

        if terminated:
            reward += 20.0

        return self.obs.copy(), reward, terminated,truncated, {"event": event, "last_shot": (r, c)}
    

    def step_eventful(self, action:int):
        obs_after, reward, terminated, truncated, info = self.step(action)
        done = terminated or truncated

        return obs_after, info.get("event", None), done

    def action_mask(self, use_length_rule=True, use_hit_focus=True):
        base = (self.obs.reshape(-1) == C_UNKNOWN)
        return base.astype(bool)
