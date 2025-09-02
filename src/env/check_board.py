#!/usr/bin/env python3
import numpy as np
from collections import Counter
from battleship_bot_with_rule import BattleshipEnv
import battleship_original as bo, importlib, os

print("USING:", bo.__file__)
print("HAS opp_address?", hasattr(bo, "opp_address"))
importlib.reload(bo)  

def check_board(arr):
    cnt = Counter(arr.reshape(-1))
    expect = {'A':5,'B':4,'D':3,'S':3,'C':2}
    for k, v in expect.items():
        assert cnt[k] == v, f"{k} count {cnt[k]} != {v}"
    return True

if __name__ == "__main__":
    env = BattleshipEnv(board_size=10, max_steps=100, seed=0)
    for _ in range(100):
        obs, _ = env.reset()
        assert check_board(env._opp), "board invalid"
    print("ok 100/100")
