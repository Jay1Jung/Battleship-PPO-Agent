# ui_pygame_min.py
# Rule-only heatmap demo: no model, no MCMC.

import pygame, numpy as np
from battleship_bot_with_rule import BattleshipEnv
from rules import RuleAgent, length_constraint, hit_technique

BOARD_SIZE = 10
CELL = 46
MARGIN = 24
BG = (245, 248, 250)
GRID = (210, 215, 220)
TXT = (35, 35, 40)
HIT = (220, 60, 60)     # assumed: hit > 0
MISS = (120, 120, 130)  # assumed: miss < 0
BEST = (60, 160, 75)

def prob_to_color(p):  # 0~1 -> 연두~빨강 느낌
    p = float(np.clip(p, 0.0, 1.0))
    r = int(60 + 195*p)
    g = int(160 - 100*p)
    b = int(75  -  35*p)
    return (r, max(g,0), max(b,0))

def rule_heatmap(obs2d: np.ndarray, remaining=None) -> np.ndarray:
    """
    룰 기반 의사 확률맵:
    - length_constraint로 놓일 수 있는 길이별 마스크에 가중치 합산
    - hit_technique가 True인 구역이 있으면 그 외는 강하게 감점
    - 이미 쏜 칸(!=0)은 배제
    """
    N = obs2d.shape[0]
    remaining = remaining or [5,4,3,3,2]
    guide = hit_technique(obs2d)               # (N,N) bool (HIT 주변 탐색 등)
    len_masks = length_constraint(obs2d, remaining)
    score = np.zeros_like(obs2d, dtype=np.float32)

    # 길이가 클수록 약간 더 가중
    for L, mask in len_masks.items():
        score[mask] += (1.0 + 0.2*L)

    # 이미 쏜 칸은 금지
    score[obs2d != 0] = -1e9

    # 가이드가 하나라도 있으면 가이드 외는 꺼버리기
    if guide.any():
        score[~guide] = -1e9

    # 음수/NaN 방지
    s = score.copy()
    s[~np.isfinite(s)] = -1e9
    return s

def draw(screen, font, obs, prob_map, best_idx):
    N = obs.shape[0]
    ox, oy = MARGIN, MARGIN+18
    screen.fill(BG)

    title = font.render("Battleship Demo (Rule-only Heatmap)", True, TXT)
    screen.blit(title, (MARGIN, 4))

    vmax = float(np.nanmax(np.where(np.isfinite(prob_map), prob_map, -1e9)))
    vmin = float(np.nanmin(np.where(np.isfinite(prob_map), prob_map, +1e9)))
    span = max(vmax - vmin, 1e-6)

    for r in range(N):
        for c in range(N):
            x, y = ox + c*CELL, oy + r*CELL
            rect = pygame.Rect(x, y, CELL-1, CELL-1)
            # 확률맵 색 칠하기 (미사일 안 쏜 칸만)
            if obs[r, c] == 0 and np.isfinite(prob_map[r, c]) and prob_map[r, c] > -1e6:
                p_norm = (prob_map[r, c] - vmin) / span
                color = prob_to_color(p_norm)
            else:
                color = (255,255,255)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID, rect, 1)

            v = obs[r, c]
            if v > 0:
                pygame.draw.circle(screen, HIT, rect.center, CELL//5)
            elif v < 0:
                pygame.draw.circle(screen, MISS, rect.center, CELL//6)

    # 베스트 칸 테두리
    if best_idx is not None:
        br, bc = divmod(int(best_idx), N)
        x, y = ox + bc*CELL, oy + br*CELL
        pygame.draw.rect(screen, (0,0,0), (x, y, CELL-1, CELL-1), 3)
        pygame.draw.rect(screen, BEST, (x+2, y+2, CELL-5, CELL-5), 3)

    hint = font.render("ENTER: shoot (recommended) | R: reset | S: screenshot | Q: quit", True, TXT)
    screen.blit(hint, (MARGIN, oy + N*CELL + 16))

def pick_best(env, ragent):
    mask = env.action_mask()
    if not mask.any():
        return None, np.zeros_like(env.obs, dtype=np.float32)
    pm = rule_heatmap(env.obs, ragent.remaining)
    # 마스크 밖 제거
    flat = pm.reshape(-1).copy()
    flat[~mask] = -1e9
    best = int(np.argmax(flat))
    return best, pm

def main():
    pygame.init()
    N = BOARD_SIZE
    screen = pygame.display.set_mode((MARGIN*2 + CELL*N, MARGIN*3 + CELL*N))
    pygame.display.set_caption("Battleship Demo (Rule-only Heatmap)")
    font = pygame.font.SysFont("Arial", 18)

    env = BattleshipEnv(board_size=N, max_steps=100)
    r = RuleAgent(board_size=N)
    obs, _ = env.reset()
    r.reset_game()

    best, pm = pick_best(env, r)
    clock, running, sn = pygame.time.Clock(), True, 0

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif e.key == pygame.K_r:
                    obs, _ = env.reset(); r.reset_game()
                    best, pm = pick_best(env, r)
                elif e.key == pygame.K_RETURN:
                    if best is not None:
                        a = int(best)
                        obs, rew, done, trunc, info = env.step(a)
                        ev = info.get("event")
                        if ev is not None:
                            rr, cc = divmod(a, env.N)
                            r.update_result((rr, cc), ev, obs)
                        if done or trunc:
                            obs, _ = env.reset(); r.reset_game()
                        best, pm = pick_best(env, r)
                elif e.key == pygame.K_s:
                    fname = f"demo_{sn:02d}.png"
                    pygame.image.save(screen, fname); sn += 1
                    print(f"Saved {fname}")

        draw(screen, font, env.obs, pm, best)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
