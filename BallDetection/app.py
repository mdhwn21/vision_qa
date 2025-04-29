import os
import tempfile
import logging
import numpy as np
import cv2
import csv
import importlib
import ball_tracker, ball_trajectory
importlib.reload(ball_tracker)
importlib.reload(ball_trajectory)
from ultralytics import YOLO
import supervision as sv
from ball_tracker import BallTracker
from ball_trajectory import BallTrajectoryAnnotator
from bounce_annotator import BounceAnnotator
from bounce_detector import BounceDetector,ConfirmBounceDetector,AngleBounceDetector
# ——— Setup absolute paths ———
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "Models")
DATA_DIR   = os.path.join(BASE_DIR, "..", "Data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "2025_v0.pt")
CSV_PATH   = os.path.join(DATA_DIR, "detections.csv")

def run_inference(
    model,
    frame: np.ndarray,
    frame_number: int,
    *,
    smoother: BallTracker,
    trajectory_annotator: BallTrajectoryAnnotator,
    angle_detector: AngleBounceDetector,
    bounce_annotator: BounceAnnotator,
    save_csv: bool = False,
    writer=None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> np.ndarray:
    if save_csv and writer is None:
        raise ValueError("writer required if save_csv=True")

    # 1) YOLO detection
    res  = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
    dets = sv.Detections.from_ultralytics(res)

    # 2) smoothing & selection
    chosen_idx, chosen_box, chosen_center = smoother.update(dets.xyxy)

    # 3) prepare for annotation
    if chosen_idx is not None:
        det_to_draw = sv.Detections(
            xyxy       = np.array([chosen_box]),
            confidence = np.array([dets.confidence[chosen_idx]]),
            class_id   = np.array([dets.class_id[chosen_idx]])
        )
    else:
        det_to_draw = dets

    # 4) draw boxes
    img = sv.BoxAnnotator().annotate(frame, det_to_draw)

    # 5) overlay frame number
    cv2.putText(
        img, str(frame_number),
        (10,30), cv2.FONT_HERSHEY_COMPLEX,
        1, (0,255,0), 2, cv2.LINE_AA
    )

    # 6) draw trajectory
    traj_str = ""
    if smoother.trajectory:
        img = trajectory_annotator.add_trajectory(img, smoother.trajectory)
        traj_str = str(list(smoother.trajectory))

    # 7) angle-based bounce detection
    traj = list(smoother.trajectory)
    bounced, angle_deg = angle_detector.detect(frame_number, traj)

    # choose center for annotation (bounce happens @ previous frame)
    if bounced and len(smoother.trajectory) >= 2:
        bounce_center = smoother.trajectory[-2]
    else:
        bounce_center = chosen_center

    img = bounce_annotator.annotate(
        img,
        bounced=bounced,
        center=bounce_center,
        frame_number=frame_number
    )

    # 8) CSV logging
    if save_csv:
        n_fields = 18
        if det_to_draw.xyxy.size == 0:
            writer.writerow([frame_number] + [""] * n_fields)
        else:
            for det_idx, (xy, conf) in enumerate(zip(det_to_draw.xyxy, det_to_draw.confidence), start=1):
                x1, y1, x2, y2 = map(float, xy)
                c_x, c_y      = (x1 + x2) / 2, (y1 + y2) / 2

                if chosen_idx is not None:
                    cx1, cy1, cx2, cy2 = map(float, chosen_box)
                    ccx, ccy = chosen_center
                    chosen_id = chosen_idx + 1
                else:
                    cx1 = cy1 = cx2 = cy2 = ccx = ccy = ""
                    chosen_id = ""

                angle_str = f"{angle_deg:.1f}" if angle_deg is not None else ""
                prev_bounce = int(bounced)

                writer.writerow([
                    frame_number,
                    det_idx,
                    conf,
                    c_x, c_y,
                    x1, y1, x2, y2,
                    chosen_id,
                    cx1, cy1, cx2, cy2,
                    ccx, ccy,
                    traj_str,
                    angle_str,
                    prev_bounce
                ])

    return img
# ——— Streamlit UI ———
import streamlit as st

st.set_page_config(page_title="Video Inference")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    logger.info("Loading model from %s", MODEL_PATH)
    return YOLO(MODEL_PATH)

model = load_model()

st.title("YOLO Video Inference App")
st.write("Upload a video and click **Run Inference** to annotate detections.")
st.write(f"Working dir: `{os.getcwd()}`")

uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if not uploaded:
    st.info("Awaiting video upload…")
    st.stop()

# Save upload to temp
if st.session_state.get("uploaded_name") != uploaded.name:
    st.session_state.uploaded_name = uploaded.name
    st.session_state.processed     = False
    suffix = os.path.splitext(uploaded.name)[1]
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    st.session_state.source_path = tmp.name
    logger.info("Saved upload to %s", tmp.name)

source = st.session_state.source_path
base, ext = os.path.splitext(os.path.basename(source))
target = os.path.join(os.path.dirname(source), f"{base}_processed{ext}")
st.session_state.target_path = target

if st.button("Run Inference"):
    with st.spinner("Running inference… this may take a while"):
        tracker = BallTracker(
            max_history=10,
            static_thresh=5,
            stable_frames=5,
            lock_frames=3,
            max_age=5)
        traj_annot = BallTrajectoryAnnotator(color=(0,255,0), thickness=2)
        angle_detector = AngleBounceDetector(min_angle_deg=40, debug=True)
        bounce_annotator = BounceAnnotator(color=(0,255,255), radius=6, thickness=1, show_frames=10)
        # Open CSV and process
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_number",
                "det_idx",
                "confidence",
                "c_x","c_y",
                "x1","y1","x2","y2",
                "chosen_id",
                "chosen_x1","chosen_y1","chosen_x2","chosen_y2",
                "chosen_cx","chosen_cy",
                "trajectory",
                "angle",
                "prev_frame_bounce"
            ])
            sv.process_video(
                source_path=source,
                target_path=target,
                callback=lambda frame, fn: run_inference(
                    model, frame, fn,
                    smoother=tracker,
                    trajectory_annotator=traj_annot,
                    angle_detector=angle_detector,
                    bounce_annotator=bounce_annotator,
                    save_csv=True,
                    writer=writer,
                    conf_threshold=0.25,
                    iou_threshold=0.45))
    

        st.session_state.processed = True
        st.success("Inference complete!")

# Download & cleanup
if st.session_state.get("processed", False):
    def cleanup():
        for key in ("source_path", "target_path"):
            p = st.session_state.get(key)
            if p and os.path.exists(p):
                try: os.remove(p)
                except OSError: logger.warning("Could not remove %s", p)
        for k in ("uploaded_name","processed","source_path","target_path"):
            st.session_state.pop(k, None)

    out_path = st.session_state.get("target_path")
    if out_path and os.path.exists(out_path):
        with open(out_path, "rb") as vid:
            st.download_button(
                "Download annotated video",
                data=vid,
                file_name=os.path.basename(out_path),
                mime="video/mp4",
                on_click=cleanup
            )
    else:
        st.warning(f"Processed video not found at `{out_path}`")

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as df:
            st.download_button(
                "Download detections CSV",
                data=df,
                file_name=os.path.basename(CSV_PATH),
                mime="text/csv"
            )
    else:
        st.warning(f"Detections CSV not found at `{CSV_PATH}`")
