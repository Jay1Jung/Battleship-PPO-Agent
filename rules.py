import numpy as np

C_UNKNOWN = 0
C_HIT = 1
C_MISS = 2
def length_constraint(obs, remaining): 

    N = obs.shape[0]
    lengths = []

    # Get empty coord length in rows
    for row in range(N):
        count = 0
        for col in range(N):
            if (obs[row,col] == C_UNKNOWN) or (obs[row,col] == C_HIT):
                count += 1
            else:
                if count >0:
                    lengths.append(count)
                    count = 0
        
        if count > 0:
            lengths.append(count)

    # Get empty coord length in cols
    for col in range(N): # transpose the 2D array
        count = 0
        for row in range(N):
            if (obs[row,col] == C_UNKNOWN) or (obs[row,col] == C_HIT):
                count += 1
        
            else:
                if count > 0:
                    lengths.append(count)
                    count = 0
                
    
        if count > 0:
            lengths.append(count)

    if remaining:  # 빈 리스트 방지
        min_rem = min(remaining)
        possible_len = sorted({L for L in lengths if L >= min_rem}, reverse=True)
        if not possible_len:
            possible_len = sorted(set(remaining), reverse=True)
    else:
        possible_len = sorted(set(lengths), reverse=True)
        

    len_mask = {L: np.zeros((N,N), dtype=bool) for L in possible_len}
    
    # check possible coord by row
    for r in range(N):
        for start_c in range(N):
            for length in possible_len:
                end_c = start_c + length

                if end_c <= N:
                    window = obs[r, start_c:end_c]
                    if np.all((window == C_UNKNOWN) | (window == C_HIT)):
                        
                        unknown_slice = (obs[r, start_c:end_c] == C_UNKNOWN)
                        len_mask[length][r, start_c:end_c] |= unknown_slice

                
    # check possible coord in col
    for c in range(N):
        for start_r in range(N):
            for L in possible_len:
                end_r = start_r + L
                
                if end_r <= N:
                    window = obs[start_r:end_r, c]
                    if np.all((window == C_UNKNOWN) | (window == C_HIT)):
                        unknown_slice = (obs[start_r:end_r, c] == C_UNKNOWN)
                        len_mask[L][start_r:end_r, c] |= unknown_slice

    return len_mask




def hit_technique(obs):

    N = obs.shape[0]
    guide_map = np.zeros((N,N), dtype = bool)
    visited = np.zeros_like(obs, dtype=bool)
    
    def valid(row, col): return 0<= row < N and 0 <= col < N

    for row in range(N):
        for col in range(N):
            if obs[row, col] != C_HIT or visited[row,col]:
                continue
                
            stack = [(row, col)]
            visited[row, col] = True
            comp = [(row, col)]

            while stack:
                r, c = stack.pop()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r + dr, c + dc
                    if valid(nr, nc) and not visited[nr, nc] and obs[nr, nc] == C_HIT:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                        comp.append((nr, nc))

            if len(comp) >= 2:
                rows = {r for r, _ in comp}
                cols = {c for _, c in comp}
                
                if len(rows) == 1:  
                    rr = next(iter(rows))
                    cs = sorted(c for _, c in comp)
                    left, right = cs[0] - 1, cs[-1] + 1
                    if valid(rr, left)  and obs[rr, left]  == C_UNKNOWN: guide_map[rr, left]  = True
                    if valid(rr, right) and obs[rr, right] == C_UNKNOWN: guide_map[rr, right] = True
                
                elif len(cols) == 1:  
                    cc = next(iter(cols))
                    rs = sorted(r for r, _ in comp)
                    up, down = rs[0] - 1, rs[-1] + 1
                    if valid(up, cc)   and obs[up, cc]   == C_UNKNOWN: guide_map[up, cc]   = True
                    if valid(down, cc) and obs[down, cc] == C_UNKNOWN: guide_map[down, cc] = True
                else:
                
                    for r, c in comp:
                        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                            nr, nc = r + dr, c + dc
                            if valid(nr, nc) and obs[nr, nc] == C_UNKNOWN:
                                guide_map[nr, nc] = True
            else:
               
                r, c = comp[0]
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r + dr, c + dc
                    if valid(nr, nc) and obs[nr, nc] == C_UNKNOWN:
                        guide_map[nr, nc] = True

    
    guide_map[obs != C_UNKNOWN] = False
    return guide_map


class RuleAgent():

    def __init__(self, board_size = 10):
        self.N = board_size
        self.reset_game()

    def reset_game(self):
        self.remaining = [5,4,3,3,2]
        self.remaining_confirmed = [5,4,3,3,2]
        self.guide_mode = False
        self.hit_cluster = []
        self.orient = None

    def select_target(self, obs):
        N = self.N
        guide = hit_technique(obs)
        has_found = bool(np.any(guide))

        len_mask = length_constraint(obs, self.remaining)
        score = np.zeros((N,N), dtype = float)

        for length, mask in len_mask.items():
            score[mask] += (1.0 + 0.1*length)
        
        score[obs != C_UNKNOWN] = 0.0

        if self.guide_mode and has_found:
            weight_map = score.copy()
            weight_map[~guide] = -1e9

            row, col = np.unravel_index(np.argmax(weight_map), weight_map.shape)

            if weight_map[row,col] > 0:
                return int(row), int(col)

            row, col = np.argwhere(guide)[0]
            return int(row), int(col) 
    
        Lmax = max(self.remaining) if self.remaining else 2
        #parity_mask = ((np.add.outer(np.arange(N), np.arange(N)) % 2) == (Lmax % 2))
        parity_black = ((np.add.outer(np.arange(N), np.arange(N)) & 1) == 0)
        parity_white = ~parity_black

        score_black = (score * parity_black).max()
        score_white = (score * parity_white).max()

        parity_mask = parity_black if score_black >= score_white else parity_white
        hunt_score = score * parity_mask
        r,c = np.unravel_index(np.argmax(hunt_score), hunt_score.shape)
        if hunt_score[r,c] <= 0:
            r,c = np.unravel_index(np.argmax(score), score.shape)
        return int(r), int(c)
    
    def update_result(self, shot_coord, result, obs_after):

        if isinstance(result, tuple) and result[0].upper() == "SINK":
            L = int(result[1])
            for lst in (self.remaining_confirmed, self.remaining):
                try: lst.remove(L)
                except ValueError: pass
            self.reset_targeting()
            return
        
        if not isinstance(result, str):
            return  
        tag = result.upper()
        
        hit_miss_map = self.get_hit_miss_map(obs_after)
        self.hit_miss_map = hit_miss_map

        if tag == "HIT":
            self.guide_mode = True
            self.hit_cluster, self.orient = self.extract_cluster_and_orient(obs_after)
            return
    
        if tag == "MISS":
            if self.guide_mode and self.hit_cluster and self.ends_blocked(self.hit_cluster, self.orient, obs_after):
                L = self.contiguous_length(self.hit_cluster)
                if L >= 2:
                    self.remove_one_remaining(L)
                self.reset_targeting()
        return
    
    def get_hit_miss_map(self,obs):
        N = obs.shape[0]
        result_map = np.full((N, N), 'O', dtype='<U1')  # 기본값 'O' = 빈칸
        
        for r in range(N):
            for c in range(N):
                if obs[r, c] == C_HIT:
                    result_map[r, c] = 'H'
                elif obs[r, c] == C_MISS:
                    result_map[r, c] = 'M'
        
        return result_map
    
    def reset_targeting(self):
        self.guide_mode = False
        self.hit_cluster = []
        self.orient = None

    def remove_one_remaining(self, length):
        try:
            self.remaining.remove(length)
        
        except ValueError: pass

    def extract_cluster_and_orient(self, obs):
        N = obs.shape[0]
        seen = np.zeros_like(obs, dtype=bool)
        best = None
        def valid(r,c): return 0<=r<N and 0<=c<N
        for row in range(N):
            for col in range(N):
                if obs[row,col] != C_HIT or seen[row,col]: 
                    continue
                stack=[(row,col)]; seen[row,col]=True; comp=[(row,col)]
                while stack:
                    r,c = stack.pop()
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=r+dr,c+dc
                        if valid(nr,nc) and not seen[nr,nc] and obs[nr,nc]==C_HIT:
                            seen[nr,nc]=True; stack.append((nr,nc)); comp.append((nr,nc))
                if best is None or len(comp) > len(best):
                    best = comp
        if not best: return [], None
        rows = {r for r,_ in best}
        cols = {c for _,c in best}
        if len(rows)==1 and len(best)>=2: orient='H'
        elif len(cols)==1 and len(best)>=2: orient='V'
        else: orient=None
        return best, orient

    def ends_blocked(self, cluster, orient, obs):

        if not cluster or orient is None: return False
        N = obs.shape[0]
        def valid(r,c): return 0<=r<N and 0<=c<N
        cells = sorted(cluster)
        if orient == 'H':
            r = next(iter({rr for rr,_ in cells}))
            cs = sorted(c for _,c in cells)
            ends = [(r, cs[0]-1), (r, cs[-1]+1)]
        else:  # 'V'
            c = next(iter({cc for _,cc in cells}))
            rs = sorted(r for r,_ in cells)
            ends = [(rs[0]-1, c), (rs[-1]+1, c)]
        blocked = 0
        for er,ec in ends:
            if not valid(er,ec): blocked += 1
            elif obs[er,ec] != C_UNKNOWN: blocked += 1   # MISS/HIT이면 확장 불가
        return blocked == 2

    def contiguous_length(self, cluster):
        if not cluster: return 0
        rows = {r for r,_ in cluster}
        cols = {c for _,c in cluster}
        if len(rows)==1:
            cs = sorted(c for _,c in cluster); return cs[-1]-cs[0]+1
        if len(cols)==1:
            rs = sorted(r for r,_ in cluster); return rs[-1]-rs[0]+1
        return 0








