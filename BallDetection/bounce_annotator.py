import cv2
from typing import Optional, Tuple

class BounceAnnotator:
    def __init__(
        self,
        color: Tuple[int,int,int] = (0, 255, 255),  # BGR yellow
        radius: int = 12,
        thickness: int = 2,
        show_frames: int = 10
    ):
        """
        color        – BGR tuple for the bounce marker
        radius       – circle radius in px
        thickness    – line thickness in px
        show_frames  – how long (in frames) to keep the marker on screen
        """
        self.color    = color
        self.radius   = radius
        self.thickness= thickness
        self.show_frames = show_frames

        self.last_bounce_frame: Optional[int]   = None
        self.last_bounce_center: Optional[Tuple[int,int]] = None

    def annotate(
        self,
        img: cv2.UMat,                            # your video frame
        bounced: bool,                            # did we just bounce?
        center: Optional[Tuple[int,int]],         # chosen_center from run_inference
        frame_number: int
    ) -> cv2.UMat:
        # 1) if we just detected a bounce, record it
        if bounced and center is not None:
            self.last_bounce_frame   = frame_number
            self.last_bounce_center  = (int(center[0]), int(center[1]))

        # 2) if we have a bounce on record and it’s still “fresh”, draw it
        if (self.last_bounce_frame is not None
            and frame_number - self.last_bounce_frame < self.show_frames
            and self.last_bounce_center is not None
        ):
            cv2.circle(
                img=img,
                center=self.last_bounce_center,
                radius=self.radius,
                color=self.color,
                thickness=-1
            )
        else:
            # 3) once it’s too old, clear it
            self.last_bounce_frame  = None
            self.last_bounce_center = None

        return img
