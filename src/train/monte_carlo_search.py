import numpy as np
from rules import length_constraint  
import multiprocessing as mp
from random import choice
import random

C_UNKNOWN = 0
C_HIT = 1
C_MISS = 2

def place_ship(board, ship_length, obs):
    N = board.shape[0]
    occ = (board == C_HIT)
    possible_positions = []

    # horizontal
    for r in range(N):
        for c in range(N - ship_length + 1):
            coords = [(r, cc) for cc in range(c, c + ship_length)]
            window_obs   = obs[r, c:c+ship_length]  
            window_occ = occ[r, c:c+ship_length]

            if np.any(window_obs == C_MISS):              
                continue
            if np.any(window_occ):         
                continue

            score = np.count_nonzero((window_obs == C_HIT) & (~window_occ))
            possible_positions.append((score, coords))


    # vertical
    for c in range(N):
        for r in range(N - ship_length + 1):
            coords = [(rr, c) for rr in range(r, r + ship_length)]
            window_obs   = obs[r:r+ship_length, c]
            window_occ = occ[r:r+ship_length, c]

            if np.any(window_obs == C_MISS):
                continue
            if np.any(window_occ):
                continue

            score = np.count_nonzero((window_obs == C_HIT) & (~window_occ))
            possible_positions.append((score, coords))

    if not possible_positions:
        return False
    
    best = max(s for s, _ in possible_positions)
    top  = [coords for s, coords in possible_positions if s == best]
    chosen = choice(top)

    for r, c in chosen:
        board[r][c] = C_HIT

    return True



def generate_random_placement_with_length_constraint(obs, remaining_ships, max_tries = 64):
    N = obs.shape[0]
    hit_coords = set(map(tuple, np.argwhere(obs == C_HIT)))
    
    for _ in range(max_tries):
        board = np.full((N, N), C_UNKNOWN, dtype=int)
        covered = set()
        ships = sorted(remaining_ships, reverse=True)
        random.shuffle(ships)

        ok = True
        for L in ships:
            candidates = []
            # horizontal
            for r in range(N):
                for c in range(N-L+1):
                    w_obs = obs[r, c:c+L]
                    if np.any(w_obs == C_MISS): continue
                    if np.any(board[r, c:c+L] == C_HIT): continue
                    covers = {(r, cc) for cc in range(c, c+L) if obs[r, cc] == C_HIT}
                    candidates.append((len(covers - covered), ('H', r, c)))
            # vertical
            for c in range(N):
                for r in range(N-L+1):
                    w_obs = obs[r:r+L, c]
                    if np.any(w_obs == C_MISS): continue
                    if np.any(board[r:r+L, c] == C_HIT): continue
                    covers = {(rr, c) for rr in range(r, r+L) if obs[rr, c] == C_HIT}
                    candidates.append((len(covers - covered), ('V', r, c)))

            if not candidates:
                ok = False; break

            best_cover = max(c[0] for c in candidates)
            pool = [meta for cov, meta in candidates if cov == best_cover] if best_cover > 0 \
                   else [meta for _, meta in candidates]

            orient, r0, c0 = random.choice(pool)
            if orient == 'H':
                for cc in range(c0, c0+L):
                    board[r0, cc] = C_HIT
                    if obs[r0, cc] == C_HIT: covered.add((r0, cc))
            else:
                for rr in range(r0, r0+L):
                    board[rr, c0] = C_HIT
                    if obs[rr, c0] == C_HIT: covered.add((rr, c0))

        if ok and hit_coords.issubset(covered):
            return board
    return None

def worker_sample(args):
    obs, remaining_ships, n_samples = args
    N = obs.shape[0]
    partial_prob_map = np.zeros((N, N), dtype=float)
    valid_samples = 0
    for _ in range(n_samples):
        sample = generate_random_placement_with_length_constraint(obs, remaining_ships)
        if sample is not None:
            partial_prob_map += (sample == C_HIT)
            valid_samples += 1
    if valid_samples > 0:
        partial_prob_map /= valid_samples
    return partial_prob_map, valid_samples


def monte_carlo_probability_map_parallel(obs, remaining_ships, num_samples, n_cores):

    if num_samples <= 0:
        return np.zeros_like(obs, dtype=np.float32)

    n_cores = max(1, min(n_cores, num_samples))

    if n_cores == 1:
        pm, vc = worker_sample((obs, remaining_ships, num_samples))
        return (pm.astype(np.float32) if vc > 0 else None)
    
    base =  num_samples // n_cores
    rem = num_samples % n_cores

    args = [(obs, remaining_ships, base + (1 if i < rem else 0))
            for i in range(n_cores) if base + (1 if i < rem else 0) > 0]


    if not args:
        pm, vc = worker_sample((obs, remaining_ships, num_samples))
        return (pm.astype(np.float32) if vc > 0 else None)
    
    with mp.Pool(processes=len(args)) as pool:
        results = pool.map(worker_sample, args)

    
    prob_map = np.zeros_like(obs, dtype=float)
    total_valid = 0
    for partial, valid in results:
        prob_map += partial * valid
        total_valid += valid

    if total_valid == 0:
        pm = np.zeros_like(obs, dtype=np.float32)
        unk = (obs == C_UNKNOWN)
        if np.any(unk):
            pm[unk] = 1.0 / np.count_nonzero(unk)
        return pm
    return (prob_map / total_valid).astype(np.float32)