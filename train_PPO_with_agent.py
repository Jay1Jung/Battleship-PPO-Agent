# train_PPO_with_rule.py
import argparse
import os
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time


from battleship_bot_with_rule import BattleshipEnv
from rules import RuleAgent, length_constraint, hit_technique

from monte_carlo_search import monte_carlo_probability_map_parallel

MCMC_TRAIN_SAMPLES = 256   
MCMC_EVAL_SAMPLES  = 2048 
MCMC_EVERY         = 4   
MCMC_CORES = int(os.getenv("MCMC_CORES", "15"))
MCMC_CORES = max(1, min(MCMC_CORES, multiprocessing.cpu_count()))

torch.set_num_threads(1)
# ---------------- Model ----------------
class PolicyValueNet(nn.Module):
    def __init__(self, n_obs=100, n_act=100):
        super().__init__()
        hid = 256
        self.fc = nn.Sequential(
            nn.Linear(n_obs, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh()
        )
        self.pi = nn.Linear(hid, n_act)
        self.v  = nn.Linear(hid, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.pi(x), self.v(x).squeeze(-1)

def flatten_obs(obs_2d):
    return obs_2d.reshape(-1).astype(np.float32)

def rule_prior_scores(obs2d: np.ndarray, remaining=None) -> np.ndarray:
    guide = hit_technique(obs2d)                           
    len_masks = length_constraint(obs2d, remaining or [5,4,3,3,2])
    score = np.zeros_like(obs2d, dtype=np.float32)
    for L, mask in len_masks.items():
        score[mask] += (1.0 + 0.2*L)
    score[obs2d != 0] = -1e9                               
    if guide.any():
        score[~guide] = -1e9                              
    s = score.reshape(-1)
    s[~np.isfinite(s)] = -1e9
    return s


@torch.no_grad()
def sample_action(net, obs_flat, action_mask, device,
                  prior_remaining=None, rule_pick=None,
                  kappa: float = 0.5, alpha: float = 1.5, greedy: bool = False):
    
    assert len(obs_flat) == action_mask.size * 2, \
    f"bad obs size: got {len(obs_flat)}, expect {action_mask.size*2}"

    x = torch.as_tensor(obs_flat, device=device).unsqueeze(0)
    logits, value = net(x)                                     
    m = torch.as_tensor(action_mask, device=device).unsqueeze(0)


    A = action_mask.shape[0]
    N = int(np.sqrt(A))
    obs_only = obs_flat[:A]

    prior_np = rule_prior_scores(obs_only.reshape(N, N), prior_remaining)
    prior = torch.as_tensor(prior_np, device=device).unsqueeze(0)

    logits = logits.masked_fill(~m, -1e9)
    prior  = prior.masked_fill(~m, -1e9)
    logits = logits + float(kappa) * prior


    if rule_pick is not None:
        boost = torch.zeros_like(logits)
        boost[0, int(rule_pick)] = float(alpha)
        logits = logits + boost

    if greedy:
        a = torch.argmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(1)).squeeze(1)
    else:
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        a = dist.sample()
        logp = dist.log_prob(a)

    return int(a.item()), float(logp.item()), float(value.item())

def compute_gae(rews, dones, vals, next_v, gamma=0.99, lam=0.95, device="cpu"):
    T = rews.shape[0]
    adv = torch.zeros(T, device=device)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = next_v if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * (1.0 - dones[t]) * v_next - vals[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    ret = adv + vals
    return adv, ret

# ---------------- PPO Train ----------------
def train(total_steps=100000, horizon=4096, lr=3e-4, device="cpu",
          seed=42, board_size=10, max_steps=100, save_path="ppo_battleship_torch.pt", mcmc_on=True,
          mcmc_train_samples=MCMC_TRAIN_SAMPLES, mcmc_every=MCMC_EVERY, mcmc_cores=MCMC_CORES ):
    torch.manual_seed(seed); np.random.seed(seed)

    env = BattleshipEnv(board_size=board_size, max_steps=max_steps, seed=seed)
    net = PolicyValueNet(n_obs= 2* env.N*env.N, n_act=env.N*env.N).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

  
    ragent = RuleAgent(board_size=board_size)

    obs, _ = env.reset()
    ragent.reset_game()  
    steps, ep_ret = 0, 0.0

    O, A, LP, V, R, D, M = [], [], [], [], [], [], []
    last_hb = time.time()

    HIT_VAL = 1  
    cached_prob_map = np.zeros_like(obs, dtype=np.float32)
    last_mcmc_step = -1
    last_hit_count = int((obs == HIT_VAL).sum())

    last_hb = time.time()
    last_steps = 0       
    last_time  = last_hb   

    while steps < total_steps:
        # ===== Rollout Collection =====
        for _ in range(horizon):

            now = time.time()
            if now - last_hb >= 10:  
                dt = max(now - last_time, 1e-8)
                sps = (steps - last_steps) / dt
                print(f"[heartbeat] steps={steps}, ep_ret={ep_ret:.2f}, "
                    f"replay_buf={len(R)}, sps={sps:.2f}", flush=True)
                last_hb    = now
                last_steps = steps    
                last_time  = now      

    
            

            need_refresh = (
                mcmc_on and (
                    (steps - last_mcmc_step >= mcmc_every) or
                    (int((obs == HIT_VAL).sum()) != last_hit_count) or
                    (cached_prob_map.shape != obs.shape) or
                    (not np.isfinite(cached_prob_map).all())
                )
            )

            if need_refresh:
                try:
                    pm = monte_carlo_probability_map_parallel(
                        obs,
                        ragent.remaining_confirmed,
                        num_samples= MCMC_TRAIN_SAMPLES,
                        n_cores=mcmc_cores,
                    )
                    if pm is None or not np.isfinite(pm).all():
                        pm = np.zeros_like(obs, dtype=np.float32)

                except Exception as e:
                    print(f"[MCMC] fallback to zeros due to error: {e}", flush=True)
                    pm = np.zeros_like(obs, dtype=np.float32)
                cached_prob_map = pm
                last_mcmc_step = steps
                last_hit_count = int((obs == HIT_VAL).sum())

            prob_map = cached_prob_map
            obs_flat = flatten_obs(obs)

            combined_input = np.concatenate([obs_flat, prob_map.reshape(-1)], dtype=np.float32)

            
            mask = env.action_mask()
            if not mask.any():
                obs, _ = env.reset()
                ragent.reset_game()
                ep_ret = 0.0
                cached_prob_map = np.zeros_like(obs, dtype=np.float32)
                last_mcmc_step = steps
                last_hit_count = int((obs == HIT_VAL).sum())
                continue

          
            rr, cc = ragent.select_target(obs)
            rule_idx = rr * env.N + cc
            if not mask[rule_idx]:
                rule_idx = None

            
            a, logp, v = sample_action(
                net, combined_input, mask, device,
                prior_remaining=ragent.remaining,
                rule_pick=rule_idx,
                kappa=0.5, alpha=1.5, greedy=False
            )

            assert mask[a], f"Picked invalid action {a} under current mask."

            obs_next, rew, done, trunc, info = env.step(a)

            
            event = info.get("event", None)  
            if event is not None:
                r_shot, c_shot = divmod(a, env.N)
                ragent.update_result((r_shot, c_shot), event, obs_next)

           
            O.append(combined_input)
            A.append(a)
            LP.append(logp)
            V.append(v)
            R.append(rew)
            D.append(float(done or trunc))
            M.append(mask.astype(bool))

            obs = obs_next
            steps += 1
            ep_ret += rew

            if done or trunc or steps >= total_steps:
                obs, _ = env.reset()
                ragent.reset_game()  
                ep_ret = 0.0
                cached_prob_map = np.zeros_like(obs, dtype=np.float32)
                last_mcmc_step = steps
                last_hit_count = int((obs == HIT_VAL).sum())
                break

        # ===== GAE/Return =====
        with torch.no_grad():
            obs_flat = flatten_obs(obs)
            obs_feat = np.concatenate([obs_flat, np.zeros_like(obs_flat, dtype=np.float32)]).astype(np.float32)
            _, next_v = net(torch.as_tensor(obs_feat, device=device).unsqueeze(0))
            next_v = next_v.squeeze(0)

        minT = min(len(R), len(D), len(V), len(A), len(LP), len(M), len(O))
        if minT == 0:
            continue

        O_t  = torch.as_tensor(np.array(O[:minT]),  dtype=torch.float32, device=device)
        A_t  = torch.as_tensor(np.array(A[:minT]),  dtype=torch.int64,   device=device)
        LP_t = torch.as_tensor(np.array(LP[:minT]), dtype=torch.float32, device=device)
        V_t  = torch.as_tensor(np.array(V[:minT]),  dtype=torch.float32, device=device)
        R_t  = torch.as_tensor(np.array(R[:minT]),  dtype=torch.float32, device=device)
        D_t  = torch.as_tensor(np.array(D[:minT]),  dtype=torch.float32, device=device)
        M_t  = torch.as_tensor(np.stack(M[:minT]),  dtype=torch.bool,    device=device)

        assert O_t.shape[0] == A_t.shape[0] == LP_t.shape[0] == V_t.shape[0] == R_t.shape[0] == D_t.shape[0] == M_t.shape[0], \
            f"rollout length mismatch: O={O_t.shape[0]} A={A_t.shape[0]} V={V_t.shape[0]} R={R_t.shape[0]} D={D_t.shape[0]} M={M_t.shape[0]}"

        ADV, RET = compute_gae(R_t, D_t, V_t, next_v, device=device)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # ===== PPO update =====
        batch = 512; clip = 0.2; epochs = 10; ent_coef = 0.01; vf_coef = 0.5; max_grad_norm = 0.5
        N = O_t.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=device)
            for st in range(0, N, batch):
                mb = perm[st:st+batch]
                mbO, mbA = O_t[mb], A_t[mb]
                mbLP, mbRET, mbADV = LP_t[mb], RET[mb], ADV[mb]
                mbM = M_t[mb]

                logits, V_now = net(mbO)
                assert mbM.shape == logits.shape, f"Mask shape mismatch: {mbM.shape} vs {logits.shape}"
                logits = logits.masked_fill(~mbM, -1e9)
                logp_all = torch.log_softmax(logits, dim=-1)
                logp_a = logp_all.gather(1, mbA.unsqueeze(1)).squeeze(1)
                ratio = (logp_a - mbLP).exp()
                clip_obj = torch.clamp(ratio, 1 - clip, 1 + clip) * mbADV
                pi_loss = -(torch.min(ratio * mbADV, clip_obj)).mean()

                v_loss = F.mse_loss(V_now, mbRET)
                ent = -(logp_all.exp() * logp_all).sum(dim=-1).mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()

        
        O.clear(); A.clear(); LP.clear(); V.clear(); R.clear(); D.clear(); M.clear()

    torch.save(net.state_dict(), save_path)
    print(f"Saved: {save_path}")
    return save_path

@torch.no_grad()
def evaluate_policy_with_rule(net, episodes=50, device="cpu", board_size=10, max_steps=100,
                              mcmc_eval_samples=MCMC_EVAL_SAMPLES, mcmc_cores=MCMC_CORES):
    env = BattleshipEnv(board_size=board_size, max_steps=max_steps)
    ragent = RuleAgent(board_size=board_size)
    wins = 0
    total_return = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ragent.reset_game()
        ep_ret, ep_steps, done = 0.0, 0, False

        while not done:
            mask = env.action_mask()
            if not mask.any():
                break


            prob_map = monte_carlo_probability_map_parallel(
                obs, ragent.remaining_confirmed,
                num_samples=mcmc_eval_samples,
                n_cores=mcmc_cores
            )
            if prob_map is None:
                prob_map = np.zeros_like(obs, dtype=np.float32)

            combined_input = np.concatenate(
                [flatten_obs(obs), prob_map.reshape(-1)]
            ).astype(np.float32)


            rr, cc = ragent.select_target(obs)
            rule_idx = rr * env.N + cc
            if not mask[rule_idx]:
                rule_idx = None

            a, _, _ = sample_action(net, combined_input, mask, device,
                                   prior_remaining=ragent.remaining,
                                   rule_pick=rule_idx,
                                   kappa=0.5, alpha=1.5, greedy=True)
            assert mask[a]
            obs, r, done, trunc, info = env.step(a)
            ep_ret += r
            ep_steps += 1
            if trunc:
                break

            event = info.get("event", None)
            if event is not None:
                r_shot, c_shot = divmod(a, env.N)
                ragent.update_result((r_shot, c_shot), event, obs)

        total_return += ep_ret
        total_steps += ep_steps
        if ep_ret >= 10.0:
            wins += 1


    return {"episodes": episodes,
            "avg_return": total_return / episodes,
            "avg_steps": total_steps / episodes,
            "win_rate": wins / episodes}

# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate_policy(net, episodes=50, device="cpu", board_size=10, max_steps=100):
    env = BattleshipEnv(board_size=board_size, max_steps=max_steps)
    wins = 0; total_return = 0.0; total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, ep_steps, done = 0.0, 0, False
        while not done:
            mask = env.action_mask()
            if not mask.any():
                break

            obs_flat = flatten_obs(obs)
            combined_input = np.concatenate(
                [obs_flat, np.zeros_like(obs_flat, dtype=np.float32)]
            ).astype(np.float32)
           
            a, _, _ = sample_action(net, combined_input, mask, device,
                        prior_remaining=None, rule_pick=None, kappa=0.0, alpha=0.0, greedy=True)
            assert mask[a]
            obs, r, done, trunc, _ = env.step(a)
            ep_ret += r; ep_steps += 1
            if trunc:
                break
        total_return += ep_ret; total_steps += ep_steps
        if ep_ret >= 10.0:
            wins += 1
        

    return {"episodes": episodes,
            "avg_return": total_return / episodes,
            "avg_steps": total_steps / episodes,
            "win_rate": wins / episodes}

@torch.no_grad()
def evaluate_random(episodes=50, board_size=10, max_steps=100, seed=123):
    env = BattleshipEnv(board_size=board_size, max_steps=max_steps)
    rng = np.random.default_rng(seed)
    wins = 0; total_return = 0.0; total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, ep_steps, done = 0.0, 0, False
        while not done:
            mask = env.action_mask()
            if not mask.any():
                break
            valid_idx = np.flatnonzero(mask)
            a = int(rng.choice(valid_idx))
            assert mask[a]
            obs, r, done, trunc, _ = env.step(a)
            ep_ret += r; ep_steps += 1
            if trunc:
                break
        total_return += ep_ret; total_steps += ep_steps
        if ep_ret >= 10.0:
            wins += 1

    return {"episodes": episodes,
            "avg_return": total_return / episodes,
            "avg_steps": total_steps / episodes,
            "win_rate": wins / episodes}

@torch.no_grad()
def evaluate_policy_with_only_rule(net, episodes=50, device="cpu", board_size=10, max_steps=100):
    """
    PPO 정책 + RuleAgent 추천만 반영 (MCMC 확률지도는 사용하지 않음).
    """
    env = BattleshipEnv(board_size=board_size, max_steps=max_steps)
    ragent = RuleAgent(board_size=board_size)
    wins = 0
    total_return = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ragent.reset_game()
        ep_ret, ep_steps, done = 0.0, 0, False

        while not done:
            mask = env.action_mask()
            if not mask.any():
                break

            # === PPO 입력 (MCMC 제외, Rule만 반영) ===
            combined_input = np.concatenate([
                flatten_obs(obs),
                np.zeros_like(obs, dtype=np.float32).reshape(-1)
            ]).astype(np.float32)

            # RuleAgent 추천 (마스크 벗어나면 무시)
            rr, cc = ragent.select_target(obs)
            rule_idx = rr * env.N + cc
            if not mask[rule_idx]:
                rule_idx = None

            # PPO의 logits + RuleAgent 추천만 boost
            a, _, _ = sample_action(net, combined_input, mask, device,
                                    prior_remaining=None,  # << MCMC Priors 안 씀!
                                    rule_pick=rule_idx,
                                    kappa=0.0, alpha=1.5, greedy=True)
            assert mask[a]

            obs, r, done, trunc, info = env.step(a)
            ep_ret += r
            ep_steps += 1
            if trunc:
                break

            # RuleAgent에 결과 반영
            event = info.get("event", None)
            if event is not None:
                r_shot, c_shot = divmod(a, env.N)
                ragent.update_result((r_shot, c_shot), event, obs)

        total_return += ep_ret
        total_steps += ep_steps
        if ep_ret >= 10.0:
            wins += 1
        
    return {
        "episodes": episodes,
        "avg_return": total_return / episodes,
        "avg_steps": total_steps / episodes,
        "win_rate": wins / episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=200_000)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--board_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--episodes_eval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="ppo_battleship_torch.pt")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--model", type=str, default="ppo_battleship_torch.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.eval_only:
        env_tmp = BattleshipEnv(board_size=args.board_size, max_steps=args.max_steps)
        net = PolicyValueNet(n_obs=2*env_tmp.N*env_tmp.N, n_act=env_tmp.N*env_tmp.N).to(device)
        net.load_state_dict(torch.load(args.model, map_location=device))
        net.eval()

        res_agent = evaluate_policy(net, episodes=args.episodes_eval, device=device,
                                    board_size=args.board_size, max_steps=args.max_steps)
        res_with_rule = evaluate_policy_with_rule(net, episodes=args.episodes_eval, device=device,
                                                board_size=args.board_size, max_steps=args.max_steps)
        res_random = evaluate_random(episodes=args.episodes_eval,
                                    board_size=args.board_size, max_steps=args.max_steps)
        res_with_only_rule = evaluate_policy_with_only_rule(net, episodes=args.episodes_eval, device=device,
                                                            board_size=args.board_size, max_steps=args.max_steps)

        print("\n=== Evaluation (PPO Agent) ===")
        print(res_agent)
        print("=== Evaluation (Random Baseline) ===")
        print(res_random)
        print("=== Evaluation (PPO_with_rule) ====")
        print(res_with_only_rule)
        print("\n=== Evaluation (PPO_with_rule and MCMC) ====")
        print(res_with_rule)
        return


    ckpt = train(args.train_steps, args.horizon, args.lr, device,
                 board_size=args.board_size, max_steps=args.max_steps,
                 save_path=args.save_path)

    env_tmp = BattleshipEnv(board_size=args.board_size, max_steps=args.max_steps)
    net = PolicyValueNet(n_obs= 2 * env_tmp.N*env_tmp.N, n_act=env_tmp.N*env_tmp.N).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    res_agent = evaluate_policy(net, episodes=args.episodes_eval, device=device,
                                board_size=args.board_size, max_steps=args.max_steps)
    res_with_rule = evaluate_policy_with_rule(net, episodes=args.episodes_eval, device=device,
                                             board_size=args.board_size, max_steps=args.max_steps)
    res_random = evaluate_random(episodes=args.episodes_eval,
                                 board_size=args.board_size, max_steps=args.max_steps)
    res_with_only_rule = evaluate_policy_with_only_rule(net, episodes=args.episodes_eval, device=device,
                                            board_size=args.board_size,max_steps=args.max_steps)

    print("\n=== Evaluation (PPO Agent) ===")
    print(res_agent)
    print("=== Evaluation (Random Baseline) ===")
    print(res_random)
    print("=== Evaluation (PPO_with_rule) ====")
    print(res_with_only_rule)
    print("\n=== Evaluation (PPO_with_rule and MCMC) ====")
    print(res_with_rule)

if __name__ == "__main__":
    main()
