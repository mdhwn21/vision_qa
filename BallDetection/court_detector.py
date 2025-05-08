import numpy as np 
import cv2
class CourtDetector:
    def __init__(self):
        self.court = None

    def set_court(self, court: np.ndarray):
        """
        Manually set the court coordinates.
        :param court: an (N×2) array of [x,y] points
        """
        self.court = court

    def detect_court(self, img: np.ndarray) -> np.ndarray:
        """
        Stub for automatic court detection logic.
        :param img: the image to run detection on
        :return: an (N×2) array of [x,y] points
        """
        raise NotImplementedError("court detection logic not implemented")

    def get_court(self, img: np.ndarray, recompute: bool = True) -> np.ndarray:
        """
        Return the cached court if available (and recompute=False), 
        otherwise run detect_court(img), cache it, and return it.
        """
        if recompute or self.court is None:
            self.court = self.detect_court(img)
        return self.court
