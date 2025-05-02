import cv2
from typing import Optional, Tuple

class BounceAnnotator:
    def __init__(
        self,
        color: Tuple[int,int,int] = (0, 0, 255),  # BGR red
        radius: int = 2,                          # smaller radius for a dot
        show_frames: int = 30
    ):
        """
        color        – BGR tuple for the bounce marker (now red)
        radius       – circle radius in px (reduced to 6 for a dot)
        show_frames  – how many frames to keep the marker on screen
        """
        self.color    = color
        self.radius   = radius
        self.show_frames = show_frames

        self.last_bounce_frame: Optional[int]  = None
        self.last_bounce_center: Optional[Tuple[int,int]] = None

    def annotate(
        self,
        img: cv2.UMat,                    # your video frame
        bounced: bool,                    # did we just bounce?
        center: Optional[Tuple[int,int]],  # chosen_center from run_inference
        frame_number: int
    ) -> cv2.UMat:
        # 1) record new bounce
        if bounced and center is not None:
            self.last_bounce_frame  = frame_number
            x, y = int(center[0]), int(center[1])
            self.last_bounce_center = (x, y)

        # 2) if bounce is still fresh, draw it as a filled red circle
        if (
            self.last_bounce_frame is not None
            and (frame_number - self.last_bounce_frame) < self.show_frames
            and self.last_bounce_center is not None
        ):
            cv2.circle(
                img=img,
                center=self.last_bounce_center,
                radius=self.radius,
                color=self.color,
                thickness=-1  # negative thickness => filled circle
            )
        else:
            # 3) once it’s too old, clear record
            self.last_bounce_frame  = None
            self.last_bounce_center = None

        return img
