import argparse
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from battleship_bot_with_rule import BattleshipEnv, C_UNKNOWN, C_HIT, C_MISS
from rules import RuleAgent  

# ---------- Model ----------
class PolicyValueNet(nn.Module):
    def __init__(self, n_obs, n_act, hid=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh()
        )
        self.pi = nn.Linear(hid, n_act)
        self.v  = nn.Linear(hid, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.pi(x), self.v(x).squeeze(-1)

@torch.no_grad()
def masked_policy(logits, mask):
    logits = logits.masked_fill(~mask, -1e9)
    probs  = F.softmax(logits, dim=-1)
    return probs

@torch.no_grad()
def pick_action(net, obs_vec, mask, device="cpu", greedy=False):
    x = torch.as_tensor(obs_vec, device=device).unsqueeze(0)
    logits, _ = net(x)
    m = torch.as_tensor(mask, device=device).unsqueeze(0)
    probs = masked_policy(logits, m)
    if greedy:
        a = int(torch.argmax(probs, dim=-1).item())
    else:
        dist = Categorical(probs=probs)
        a = int(dist.sample().item())
    return a, probs.squeeze(0).cpu().numpy()

def flatten_obs(obs_2d):
    return obs_2d.reshape(-1).astype(np.float32)

# ---------- Rendering ----------
def coord_str(r, c):
    return f"{chr(ord('A')+r)}{c+1}"

def parse_coord(s, N):
    s = s.strip().upper()
    m = re.fullmatch(r"([A-Z])\s*(\d+)", s)
    if m:
        r = ord(m.group(1)) - ord('A')
        c = int(m.group(2)) - 1
        if 0 <= r < N and 0 <= c < N:
            return r, c
        return None
    m = re.fullmatch(r"(\d+)[ ,]+(\d+)", s)
    if m:
        r = int(m.group(1)); c = int(m.group(2))
        if 0 <= r < N and 0 <= c < N:
            return r, c
        return None
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < N*N:
            return idx // N, idx % N
    return None

def render_board(env, reveal=False):
    N = env.N
    obs = env.obs
    header = "    " + " ".join(f"{i+1:2d}" for i in range(N))
    print(header)
    for r in range(N):
        row_label = chr(ord('A')+r)
        cells = []
        for c in range(N):
            v = obs[r, c]
            ch = "."
            if v == C_HIT:  ch = "X"
            elif v == C_MISS: ch = "o"
            if reveal and hasattr(env, "_opp"):
                opp = env._opp[r, c]
                if v == C_UNKNOWN and opp in {"A","B","C","D","S"}:
                    ch = opp.lower()
            cells.append(ch)
        print(f"{row_label} | " + " ".join(f"{c:2s}" for c in cells))

def show_topk(probs, mask, N, k=5):
    valid_idx = np.flatnonzero(mask)
    if valid_idx.size == 0:
        print("  (no valid actions)")
        return
    p = probs.copy()
    p[~mask] = -1.0
    order = np.argsort(-p)
    out = []
    for i in order:
        if not mask[i]: continue
        if len(out) >= k: break
        r, c = divmod(i, N)
        out.append((i, p[i], coord_str(r, c)))
    print("  Top-{} suggestions:".format(k))
    for idx, prob, s in out:
        print(f"   - {s:>4s} (idx={idx:3d})  p={prob:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["human","ppo","random"], default="human")
    ap.add_argument("--model", type=str, default="ppo_battleship_torch.pt")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--board_size", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = BattleshipEnv(board_size=args.board_size, max_steps=args.max_steps, seed=args.seed)

    net = None
    if args.mode == "ppo" or (args.mode == "human" and args.model):
        n = env.N * env.N
        net = PolicyValueNet(n_obs= 2*n, n_act=n).to(device)
        try:
            sd = torch.load(args.model, map_location=device)
            net.load_state_dict(sd)
            net.eval()
            print(f"[loaded model] {args.model}")
        except Exception as e:
            if args.mode == "ppo":
                print(f"!! failed to load model for PPO mode: {e}")
                sys.exit(1)
            else:
                print(f"(!) hint/model not loaded: {e}")

    obs, _ = env.reset()
    done, steps, ep_ret = False, 0, 0.0

    ragent = RuleAgent(board_size=args.board_size)
    ragent.reset_game()

    print("=== Battleship: play ===")
    print("symbols: .(unknown)  X(hit)  o(miss)")
    if args.mode == "human":
        print("input examples:  A5   or   0 4   or   23")
        print("commands:  rand  |  hint  |  quit")

    while not done:
        print()
        render_board(env, reveal=args.reveal)
        mask = env.action_mask(use_length_rule=True, use_hit_focus=True)
        valid_idx = np.flatnonzero(mask)
        print(f"[step {steps}] valid actions: {len(valid_idx)}")
        if len(valid_idx) == 0:
            print("No valid actions; ending.")
            break

        if args.mode == "human":
            probs = None
            if net is not None:
                _, probs = pick_action(net, flatten_obs(obs), mask, device=device, greedy=False)
            while True:
                s = input("> choose / rand / hint / quit: ").strip()
                if s.lower() in ("quit","q","exit"):
                    return
                if s.lower() in ("rand","r"):
                    a = int(np.random.choice(valid_idx))
                    break
                if s.lower().startswith("hint"):
                    if probs is None:
                        print("  (no model loaded for hint)")
                    else:
                        show_topk(probs, mask, env.N, k=args.topk)
                    continue
                rc = parse_coord(s, env.N)
                if rc is None:
                    print("  invalid input.")
                    continue
                r, c = rc
                a = r * env.N + c
                if not mask[a]:
                    print("  not allowed by rules.")
                    continue
                break

        elif args.mode == "ppo":
            
            a, probs = pick_action(net, flatten_obs(obs), mask, device=device, greedy=args.greedy)
            rr, cc = ragent.select_target(obs)
            rule_a = rr * env.N + cc
            if mask[rule_a]:
                a = rule_a
            r, c = divmod(a, env.N)
            print(f"agent -> {coord_str(r,c)} (idx={a})")

        else: 
            a = int(np.random.choice(valid_idx))
            r, c = divmod(a, env.N)
            print(f"random -> {coord_str(r,c)} (idx={a})")

        r0, c0 = divmod(a, env.N)
        obs, rew, done, trunc, info = env.step(a)
        ep_ret += rew
        steps += 1
        outcome = "HIT" if env.obs[r0, c0] == C_HIT else "MISS"
        print(f"shot {coord_str(r0,c0)}  =>  {outcome}   reward={rew:+.2f}   total={ep_ret:.2f}")

       
        event = info.get("event", None)
        if event is not None:
            ragent.update_result((r0, c0), event, obs)

        if trunc:
            print("Truncated by env (max_steps).")
            break

    print("\n=== Episode end ===")
    print(f"steps={steps}   return={ep_ret:.2f}")
    print("Good game!")

if __name__ == "__main__":
    main()
