from collections import deque
import numpy as np
import cv2
class BounceAnnotator:
    def __init__(
        self,
        color: tuple[int,int,int] = (0, 0, 255),  # BGR red
        radius: int = 15,
        thickness: int = 2
    ):
        self.color     = color
        self.radius    = radius
        self.thickness = thickness

    def annotate(
        self,
        img: np.ndarray,
        center: tuple[int,int]
    ) -> np.ndarray:
        """
        Draw a circle at the detected bounce center.
        """
        cv2.circle(
            img=img,
            center=(int(center[0]), int(center[1])),
            radius=self.radius,
            color=self.color,
            thickness=self.thickness
        )
        return img
