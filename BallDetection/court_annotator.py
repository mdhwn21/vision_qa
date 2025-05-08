import numpy as np
import cv2

class CourtAnnotator:
    def __init__(self):
        # Original polygon as an Nx2 float array
        self.orig_court   = None  
        # OpenCV contours (Nx1x2 int32)
        self.court        = None  
        self.court_small  = None  
        self.court_big    = None  
        # Mask of the “ring” between big & small
        self.mask         = None  

    @staticmethod
    def to_cv_contour(pts: np.ndarray) -> np.ndarray:
        """
        Turn an (N,2) array into an OpenCV contour of shape (N,1,2) with dtype int32.
        """
        return pts.reshape(-1, 1, 2).astype(np.int32)

    def set_court(self, court_pts: np.ndarray):
        """
        Manually set the court polygon.
        :param court_pts: an (N,2) array of [x,y] points (float or int).
        """
        self.orig_court = np.array(court_pts, dtype=np.float32)
        self.court = self.to_cv_contour(self.orig_court.astype(np.int32))

    def detect_court(self, img: np.ndarray) -> np.ndarray:
        """
        Stub for automatic court detection logic.
        :return: an (N,2) float32 array of detected points.
        """
        raise NotImplementedError("court detection logic not implemented")

    def get_court(self, img: np.ndarray, recompute: bool = False) -> np.ndarray:
        """
        Return the cached court if available (and recompute=False),
        otherwise detect it, cache it, and return it.
        """
        if recompute or self.orig_court is None:
            pts = self.detect_court(img)
            self.set_court(pts)
        return self.orig_court

    def set_buffer_courts(self, img: np.ndarray, scale: float = 0.1):
        """
        Build  ±scale buffers around the court and create a mask
        of the ring between them.
        """
        if self.orig_court is None:
            raise ValueError("Court must be set before buffering")

        pts      = self.orig_court
        centroid = pts.mean(axis=0)

        big_pts   = centroid + (pts - centroid) * (1 + scale)
        small_pts = centroid + (pts - centroid) * (1 - scale)

        # save as cv2‐contours
        self.court_big   = self.to_cv_contour(big_pts.astype(np.int32))
        self.court_small = self.to_cv_contour(small_pts.astype(np.int32))

        # build masks
        mask_big   = np.zeros(img.shape[:2], dtype=np.uint8)
        mask_small = np.zeros_like(mask_big)
        cv2.fillPoly(mask_big,   [self.court_big],   255)
        cv2.fillPoly(mask_small, [self.court_small], 255)

        # ring = large minus small
        self.mask = cv2.bitwise_and(mask_big, cv2.bitwise_not(mask_small))

    def annotate(self, img: np.ndarray,
                 alpha: float = 0.3,
                 apply_buffer: bool = True) -> np.ndarray:
        """
        Draw the court and optional buffer‐zone overlay.
        :return: annotated copy of img
        """
        if self.court is None:
            return img  # nothing to do

        out = img.copy()
        # draw main court in yellow
        cv2.polylines(out, [self.court], True, (0,255,255), 2)

        if apply_buffer:
            # overlay the buffer ring in red, semi‐transparent
            overlay = out.copy()
            overlay[self.mask == 255] = (0,0,255)
            out = cv2.addWeighted(overlay, alpha, out, 1-alpha, 0)

            # optionally outline the buffers
            cv2.polylines(out, [self.court_big],   True, (0,255,0), 2)
            cv2.polylines(out, [self.court_small], True, (255,0,0), 2)

        return out
