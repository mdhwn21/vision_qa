from collections import deque
import numpy as np

class BallTracker:
    def __init__(
        self,
        max_history: int = 10,
        static_thresh: float = 5.0,
        stable_frames: int = 5,
        lock_frames: int = 3,
        max_age: int = 5,
        miss_thresh: int = 5,
        jump_thresh: float = 100.0
    ):
        # existing parameters
        self.max_history    = max_history
        self.static_thresh  = static_thresh
        self.stable_frames  = stable_frames
        self.lock_frames    = lock_frames
        self.max_age        = max_age

        # new reset parameters
        self.miss_thresh    = miss_thresh    # reset after this many misses
        self.jump_thresh    = jump_thresh    # reset if jump > this many px

        # per-candidate histories / ages
        self.histories      = {}             # cid -> deque[(cx,cy)]
        self.ages           = {}             # cid -> frames since last match

        # locked‐ball state
        self.locked_id      = None
        self.locked_count   = 0
        self.last_center    = None

        # our single‐ball trajectory
        self.trajectory     = deque(maxlen=self.max_history)

        # count of consecutive frames with no chosen_center
        self.miss_count     = 0

    def reset_all(self):
        """Clear trajectory and unlock everything."""
        self.locked_id    = None
        self.locked_count = 0
        self.last_center  = None
        self.histories.clear()
        self.ages.clear()
        self.trajectory.clear()
        self.miss_count   = 0

    def update(self, xyxy: np.ndarray):
        """
        xyxy: (N,4) array of [x1,y1,x2,y2]
        Returns: (chosen_cid, chosen_box, chosen_center)
        """
        # --- detect single-box quick-lock ---
        if self.locked_id is None and xyxy.shape[0] == 1:
            x1, y1, x2, y2 = xyxy[0]
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            self.reset_all()
            # seed candidate 0
            self.histories[0] = deque(maxlen=self.max_history)
            self.histories[0].append((cx, cy))
            self.ages[0]      = 0
            self.locked_id    = 0
            self.last_center  = (cx, cy)
            # reset trajectory, then record the first point
            self.trajectory.append((cx, cy))
            return 0, (x1, y1, x2, y2), (cx, cy)

        # --- build centers list ---
        centers = [((x1+x2)/2,(y1+y2)/2) for x1,y1,x2,y2 in xyxy]

        # 1) age‐out old cids
        for cid in list(self.ages):
            self.ages[cid] += 1
            if self.ages[cid] > self.max_age:
                del self.ages[cid], self.histories[cid]

        # 2) match or spawn new cids
        new_ages = {}
        for c in centers:
            cid, last = min(
                ((cid, hist[-1]) for cid, hist in self.histories.items()),
                key=lambda kv: np.hypot(kv[1][0]-c[0], kv[1][1]-c[1]),
                default=(None, None)
            )
            if cid is not None and np.hypot(last[0]-c[0], last[1]-c[1]) < self.static_thresh:
                self.histories[cid].append(c)
                new_ages[cid] = 0
            else:
                new_cid = max(self.histories.keys(), default=-1) + 1
                self.histories[new_cid] = deque(maxlen=self.max_history)
                self.histories[new_cid].append(c)
                new_ages[new_cid] = 0
        self.ages = new_ages

        # 3) static‐filter & lock‐in logic
        if self.locked_id is None:
            movers = [
                cid for cid, hist in self.histories.items()
                if len(hist) >= self.stable_frames
                and np.hypot(hist[-1][0]-hist[0][0], hist[-1][1]-hist[0][1]) >= self.static_thresh
            ]
            if len(movers) == 1:
                self.locked_count += 1
                if self.locked_count >= self.lock_frames:
                    self.locked_id    = movers[0]
                    self.last_center  = self.histories[self.locked_id][-1]
            else:
                self.locked_count = 0

        # 4) steady‐state: choose by nearest to last_center
        if self.locked_id is not None:
            if not centers:
                # miss: no detection this frame
                self.miss_count += 1
                if self.miss_count >= self.miss_thresh:
                    self.reset_all()
                return None, None, None

            # we got a detection, reset miss counter
            self.miss_count = 0

            # pick nearest
            dists = [
                (i, np.hypot(cx-self.last_center[0], cy-self.last_center[1]))
                for i,(cx,cy) in enumerate(centers)
            ]
            chosen_idx, dist = min(dists, key=lambda t: t[1])
            chosen_box    = xyxy[chosen_idx]
            chosen_center = centers[chosen_idx]

            # sudden‐jump reset?
            if self.trajectory and dist > self.jump_thresh:
                # clear and seed with this outlier as fresh start
                self.trajectory.clear()

            # record
            self.last_center = chosen_center
            self.trajectory.append(chosen_center)

            return chosen_idx, chosen_box, chosen_center

        # if still not locked, do nothing
        return None, None, None
