# First you’ll only enter FALLING if you see at least 2 px/frame downward (dy > 2).
# After at least 2 frames of that, you watch for dy to slow below 8 px/frame—on that very frame you fire the bounce, even though the ball may still be moving downward in pixel-space.

# dy_thresh: float = 2.0
# What it is: the minimum downward speed (in pixels/frame) we require before we’ll consider the ball to be in a true “fall.”
# Role: guards against tiny jitters being mistaken for the start of a bounce cycle. Only when your smoothed Δy exceeds this (i.e. you’re dropping at more than 2 px/frame) do we flip to the FALLING state.

# Tuning: raise it if you still catch small camera or detection noise; lower it if your ball starts slower.
# low_vel_thresh: float = 8.0
# What it is: the maximum downward speed we will accept as the “bottom” of the bounce.
# Role: once in FALLING, we watch for Δy to drop below this threshold (even if still positive)—that slowing‐point is our bounce trigger.

# Tuning: you measured ~5–6 px/frame at impact, so setting this around 8 px/frame ensures we fire exactly when the descent noticeably decelerates.

# min_fall_frames: int = 2
# What it is: the number of consecutive frames your smoothed Δy must stay above dy_thresh before you’re allowed to declare a bounce.
# Role: prevents single‐frame spikes from triggering a false cycle—only after you’ve truly been “falling” for at least 2 frames will you then watch for that velocity-drop to fire a bounce.
# Tuning: increase to 3–4 if you see brief two-frame jitters; keep it low to catch quick bounces.

# smooth_frames: int = 3
# What it is: the window size (in frames) for your moving‐average of Y-positions.
# Role: replaces raw yₜ – yₜ₋₁ with (avg of last 3 y’s) – (avg of prior 3 y’s), smoothing out single‐frame noise so dy more accurately reflects true motion.

# Tuning:
# Smaller (2–3) keeps you more responsive but less noise-free.
# Larger (5–7) gives smoother velocities at the cost of a slight lag in detection.

# All dy’s are computed on a 3-frame average of Y, so your detector isn’t fooled by jittery center estimates.
# from collections import deque
# import numpy as np
# from typing import Optional, Tuple

from collections import deque
import numpy as np
from typing import Optional, Tuple

class BounceDetector:
    def __init__(
        self,
        smooth_frames: int = 3,         # window for smoothing Y
        accel_thresh: float = 10.0,     # px/frame² threshold to fire bounce
        debug: bool = False
    ):
        self.smooth_frames = smooth_frames
        self.accel_thresh  = accel_thresh
        self.debug         = debug

        # buffers / history
        self.y_buffer       = deque(maxlen=smooth_frames)
        self.prev_smooth_y  : Optional[float] = None
        self.prev_dy        : Optional[float] = None
        self.prev_ddy_sign  : Optional[int]   = None

        # state
        self.armed          = False   # True once we see a sign-flip in ddy
        self.bounce_ready   = True    # blocks further bounces until re-enabled

    def feed(
        self,
        center: Optional[Tuple[float, float]],
        frame_number: Optional[int] = None
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        if center is None:
            # reset everything
            self.y_buffer.clear()
            self.prev_smooth_y = self.prev_dy = self.prev_ddy_sign = None
            self.armed = False
            self.bounce_ready = True
            return False, None, None

        _, raw_y = center
        self.y_buffer.append(raw_y)
        smooth_y = float(np.mean(self.y_buffer))

        # first frame: init only
        if self.prev_smooth_y is None:
            self.prev_smooth_y = smooth_y
            return False, None, None

        # compute velocity & acceleration
        dy = smooth_y - self.prev_smooth_y
        ddy = None
        if self.prev_dy is not None:
            ddy = dy - self.prev_dy

        if self.debug:
            fn = f"[F{frame_number}] " if frame_number is not None else ""
            print(fn + f"dy={dy:.1f}, ddy={ddy if ddy is None else f'{ddy:.1f}'} "
                        f"armed={self.armed} ready={self.bounce_ready}")

        bounce = False

        # once we have an acceleration sample
        if ddy is not None:
            sig = int(np.sign(ddy))
            # first accel→ just record
            if self.prev_ddy_sign is None:
                self.prev_ddy_sign = sig
            else:
                # sign-flip!
                if sig != self.prev_ddy_sign:
                    if self.bounce_ready:
                        # arm for bounce on next big |ddy|
                        self.armed = True
                        if self.debug:
                            print(fn + "→ accel sign-flip: armed=True")
                    else:
                        # we've already bounced; this flip re-enables detection
                        self.bounce_ready = True
                        if self.debug:
                            print(fn + "→ accel sign-flip: bounce_ready=True")
                    self.prev_ddy_sign = sig

            # if armed and threshold exceeded, fire bounce
            if self.armed and self.bounce_ready and abs(ddy) >= self.accel_thresh:
                bounce = True
                self.armed = False      # disarm until next re-enable
                self.bounce_ready = False
                if self.debug:
                    print(fn + f"** BOUNCE (|ddy|={abs(ddy):.1f} ≥ {self.accel_thresh}) **")

        # shift history for next frame
        self.prev_smooth_y = smooth_y
        self.prev_dy       = dy

        return bounce, dy, ddy


from collections import deque
import numpy as np
from typing import Optional, Tuple

class ConfirmBounceDetector:
    def __init__(
        self,
        smooth_frames: int = 3,      # window for Y smoothing
        confirm_frames: int = 2,     # how many frames to confirm the flip
        debug: bool = False
    ):
        self.smooth_frames  = smooth_frames
        self.confirm_frames = confirm_frames
        self.debug          = debug

        # smoothing buffer
        self.y_buffer       = deque(maxlen=smooth_frames)
        self.prev_smooth_y  : Optional[float] = None

        # velocity history
        self.prev_dy        : Optional[float] = None

        # for detecting flips
        self.prev_ddy_sign  : Optional[int] = None
        self.candidate_sign : Optional[int] = None
        self.confirm_count  : int = 0

    def feed(
        self,
        center: Optional[Tuple[float, float]],
        frame_number: Optional[int] = None
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Returns:
          bounce_detected: bool
          dy:  smoothed vertical velocity (px/frame)
          ddy: acceleration (dy - prev_dy), or None on first sample
        """
        if center is None:
            # full reset
            self.y_buffer.clear()
            self.prev_smooth_y  = None
            self.prev_dy        = None
            self.prev_ddy_sign  = None
            self.candidate_sign = None
            self.confirm_count  = 0
            return False, None, None

        _, raw_y = center
        self.y_buffer.append(raw_y)
        smooth_y = float(np.mean(self.y_buffer))

        # init
        if self.prev_smooth_y is None:
            self.prev_smooth_y = smooth_y
            return False, None, None

        # compute dy and ddy
        dy = smooth_y - self.prev_smooth_y
        ddy = None
        if self.prev_dy is not None:
            ddy = dy - self.prev_dy

        if self.debug:
            fn = f"[F{frame_number}] " if frame_number is not None else ""
            print(fn + f"dy={dy:.1f}, ddy={ddy if ddy is None else f'{ddy:.1f}'} "
                        f"cand_sign={self.candidate_sign} confirms={self.confirm_count}")

        bounce = False

        if ddy is not None:
            curr_sign = int(np.sign(ddy))

            # 1) new flip → start a candidate
            if self.prev_ddy_sign is not None and curr_sign != self.prev_ddy_sign:
                self.candidate_sign = curr_sign
                self.confirm_count  = 0
                if self.debug:
                    print(fn + f"→ sign-flip detected; candidate_sign={curr_sign}")

            # 2) if we have a candidate, confirm it
            elif self.candidate_sign is not None:
                if curr_sign == self.candidate_sign:
                    self.confirm_count += 1
                    if self.debug:
                        print(fn + f"→ confirm {self.confirm_count}/{self.confirm_frames}")
                else:
                    # sign reversed again or noisy → drop candidate
                    if self.debug:
                        print(fn + "→ confirmation failed; discarding candidate")
                    self.candidate_sign = None
                    self.confirm_count  = 0

                # 3) once confirmed enough, fire bounce
                if self.confirm_count >= self.confirm_frames:
                    bounce = True
                    if self.debug:
                        print(fn + "** BOUNCE confirmed **")
                    # reset for next detection
                    self.candidate_sign = None
                    self.confirm_count  = 0

            # update previous sign
            self.prev_ddy_sign = curr_sign

        # shift for next frame
        self.prev_smooth_y = smooth_y
        self.prev_dy       = dy

        return bounce, dy, ddy

import numpy as np
from typing import List, Tuple, Optional
import cv2
import supervision as sv

class AngleBounceDetector:
    def __init__(
        self,
        min_angle_deg: float = 10.0,   # threshold in degrees
        debug: bool = False
    ):
        self.min_angle = np.deg2rad(min_angle_deg)
        self.debug     = debug

    def detect(
        self,
        frame_number: int,
        trajectory: List[Tuple[float, float]]
    ) -> Tuple[bool, Optional[float]]:
        """
        Online bounce detection using only the last three points.
        Returns:
          bounced: True if turn-angle at middle point ≥ threshold
          angle_deg: the measured angle in degrees (or None if not computable)
        """
        # need at least 3 points
        if len(trajectory) < 3:
            return False, None

        # take last 3 centers
        (x0, y0), (x1, y1), (x2, y2) = trajectory[-3:]

        # build vectors
        v1 = np.array([x1 - x0, y1 - y0])
        v2 = np.array([x2 - x1, y2 - y1])

        # compute norms and skip degenerate
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return False, None

        # compute angle
        cosang = np.dot(v1, v2) / (norm1 * norm2)
        ang_rad = np.arccos(np.clip(cosang, -1.0, 1.0))
        angle_deg = np.rad2deg(ang_rad)

        if self.debug:
            prefix = f"[F{frame_number}] "
            print(f"{prefix}Turn angle at middle = {angle_deg:.1f}°")

        bounced = (ang_rad >= self.min_angle)
        return bounced, angle_deg
