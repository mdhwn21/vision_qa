# ——— your BallTrajectoryAnnotator 

import cv2
import numpy as np
from typing import Sequence, Tuple

class BallTrajectoryAnnotator:
    def __init__(
        self,
        color: Tuple[int,int,int] = (0, 255, 0),
        thickness: int = 2,
        min_points: int = 2
    ):
        """
        color      – BGR tuple for the trajectory line
        thickness  – line thickness in px
        min_points – only draw if we have at least this many centers
        """
        self.color = color
        self.thickness = thickness
        self.min_points = min_points

    def add_trajectory(
        self,
        img: np.ndarray,
        centers: Sequence[Tuple[int,int]]
    ) -> np.ndarray:
        """
        Draws the trajectory of the ball on `img` given its history of centers.

        Args:
          img     : HxWx3 BGR image to draw on (will be modified in-place)
          centers : list or deque of (x, y) int tuples

        Returns:
          The same `img`, with the polyline drawn if len(centers) >= min_points.
        """
        if len(centers) >= self.min_points:
            pts = np.array(centers, dtype=np.int32)
            cv2.polylines(
                img,
                [pts],
                isClosed=False,
                color=self.color,
                thickness=self.thickness
            )
        return img