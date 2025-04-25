# ——— your BallTracker class (from previous message) ———
from collections import deque
import numpy as np

class BallTracker:
    def __init__(self,
                 max_history: int = 10,
                 static_thresh: float = 5.0,
                 stable_frames: int = 5,
                 lock_frames: int = 3,
                 max_age: int = 5):
        self.max_history      = max_history
        self.static_thresh    = static_thresh
        self.stable_frames    = stable_frames
        self.lock_frames      = lock_frames
        self.max_age          = max_age

        # per-candidate histories and ages
        self.histories        = {}            # cid -> deque[(cx,cy)]
        self.ages             = {}            # cid -> frames since last match

        self.locked_id        = None          # once decided, this is the cid
        self.locked_count     = 0             # how many frames in a row exactly one mover
        self.last_center      = None
        self.trajectory = deque(maxlen=self.max_history)

    def update(self, xyxy: np.ndarray):
        """
        xyxy: (N,4) array of [x1,y1,x2,y2]
        Returns: (chosen_cid, chosen_box, chosen_center)
                 or (None, None, None) if not locked yet
        """
        # ——— Early-out for single detection ———
        if self.locked_id is None and xyxy.shape[0] == 1:
            # immediately lock on candidate 0
            x1, y1, x2, y2 = xyxy[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # reset history to just this one candidate
            self.histories.clear()
            self.ages.clear()
            self.histories[0]   = deque(maxlen=self.max_history)
            self.histories[0].append((cx, cy))
            self.ages[0]        = 0
            self.locked_id      = 0
            self.last_center    = (cx, cy)
            # ***append to your single-ball trajectory***
            self.trajectory.clear()            # start fresh
            self.trajectory.append((cx, cy))
            return 0, (x1, y1, x2, y2), (cx, cy)
        
        
        
        centers = [((x1+x2)/2,(y1+y2)/2) for x1,y1,x2,y2 in xyxy]

        # 1) age out
        for cid in list(self.ages):
            self.ages[cid] += 1
            if self.ages[cid] > self.max_age:
                del self.ages[cid], self.histories[cid]

        # 2) match or spawn
        new_ages = {}
        for c in centers:
            # find nearest existing cid
            best = min(
                ((cid, hist[-1]) for cid, hist in self.histories.items()),
                key=lambda kv: np.hypot(kv[1][0]-c[0], kv[1][1]-c[1]),
                default=(None, None)
            )
            cid, last = best
            if cid is not None and np.hypot(last[0]-c[0], last[1]-c[1]) < self.static_thresh:
                # matched
                self.histories[cid].append(c)
                new_ages[cid] = 0
            else:
                # new candidate
                new_cid = max(self.histories.keys(), default=-1) + 1
                self.histories[new_cid] = deque(maxlen=self.max_history)
                self.histories[new_cid].append(c)
                new_ages[new_cid] = 0

        self.ages = new_ages

        # 3) static filter & lock-in
        if self.locked_id is None:
            movers = [
                cid for cid, hist in self.histories.items()
                if len(hist) >= self.stable_frames and
                   np.hypot(hist[-1][0]-hist[0][0], hist[-1][1]-hist[0][1]) >= self.static_thresh
            ]
            if len(movers) == 1:
                self.locked_count += 1
                if self.locked_count >= self.lock_frames:
                    self.locked_id   = movers[0]
                    self.last_center = self.histories[self.locked_id][-1]
            else:
                self.locked_count = 0

        # 4) steady-state: pick closest to last_center
        if self.locked_id is not None:
            # if there are no detections, just return None
            if len(centers) == 0:
                return None, None, None
    
            # build distance list (guaranteed non-empty now)
            dists = [
                (i, np.hypot(cx - self.last_center[0], cy - self.last_center[1]))
                for i, (cx, cy) in enumerate(centers)
            ]
            chosen_idx, _    = min(dists, key=lambda t: t[1])
            chosen_box       = xyxy[chosen_idx]
            chosen_center    = centers[chosen_idx]
            self.last_center = chosen_center
            if chosen_center is not None:
                # ***append chosen_center to your single‐ball trajectory***
                self.trajectory.append(chosen_center)
            return chosen_idx, chosen_box, chosen_center
        return None, None, None