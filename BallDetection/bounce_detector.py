from collections import deque
import numpy as np
import cv2

class BounceDetector:
    def __init__(
        self,
        dy_thresh: float = 2.0,
        min_fall_frames: int = 2,
        history_len: int = 10
    ):
        """
        dy_thresh      – minimum |Δy| (px) to count as significant motion
        min_fall_frames– require ≥ this many consecutive falling frames
        history_len    – length of the internal deque (not strictly needed)
        """
        self.dy_thresh       = dy_thresh
        self.min_fall_frames = min_fall_frames

        # last‐Y to compute Δy
        self.prev_y    = None
        # how many frames we’ve seen falling in a row
        self.fall_count= 0
        # state machine: IDLE → RISING → FALLING → (bounce) → IDLE
        self.state     = "IDLE"

    def feed(self, center: tuple[float, float]) -> tuple[bool, float|None]:
        """
        Push the latest (cx, cy). Returns:
          bounce: bool            – True only on the frame we detect a bounce
          dy:     float or None   – most recent vertical velocity (y_t - y_{t-1})
        """
        # no center → reset to IDLE
        if center is None:
            self.state      = "IDLE"
            self.fall_count = 0
            self.prev_y     = None
            return False, None

        _, y = center

        # first frame ever: just store and exit
        if self.prev_y is None:
            self.prev_y = y
            return False, None

        # compute vertical velocity
        dy = y - self.prev_y
        self.prev_y = y

        bounce = False

        if self.state == "IDLE":
            if dy < -self.dy_thresh:
                # moving up (pixel‐coords: smaller y)
                self.state = "RISING"
            elif dy > self.dy_thresh:
                # moving down
                self.state      = "FALLING"
                self.fall_count = 1

        elif self.state == "RISING":
            if dy > self.dy_thresh:
                # topped out, now falling
                self.state      = "FALLING"
                self.fall_count = 1

        elif self.state == "FALLING":
            if dy > self.dy_thresh:
                # still falling
                self.fall_count += 1
            elif dy < 0:
                # sign‐flip: we were falling then start rising → bounce
                if self.fall_count >= self.min_fall_frames:
                    bounce = True
                # reset state for next detection
                self.state      = "IDLE"
                self.fall_count = 0

        return bounce, dy