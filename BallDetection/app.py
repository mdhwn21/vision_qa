import streamlit as st
import os
import tempfile
import logging
from ultralytics import YOLO
import supervision as sv

# ——— Page & Logger Setup ———
st.set_page_config(page_title="Video Inference")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Load & Cache Model ———
@st.cache_resource
def load_model():
    logger.info("Loading model")
    return YOLO("../Models/2025_v0.pt")

model = load_model()

# ——— UI ———
st.title("YOLO Video Inference App")
st.write("Upload a video and click **Run Inference** to annotate detections.")

uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded:
    # Detect a new upload and reset state
    if st.session_state.get("uploaded_name") != uploaded.name:
        st.session_state.uploaded_name = uploaded.name
        st.session_state.processed     = False
        # Write upload to temp file
        suffix = os.path.splitext(uploaded.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state.source_path = tmp.name
        logger.info(f"Saved upload to {tmp.name}")

    # Prepare target path
    source = st.session_state.source_path
    base, ext = os.path.splitext(os.path.basename(source))
    target = os.path.join(os.path.dirname(source), f"{base}_processed{ext}")
    st.session_state.target_path = target

    # Run inference only when button is clicked
    if st.button("Run Inference"):
        with st.spinner("Running inference… this may take a while"):
            sv.process_video(
                source_path=source,
                target_path=target,
                callback=lambda frame, frame_number: sv.BoxAnnotator().annotate(
                    frame,
                    sv.Detections.from_ultralytics(model(frame)[0])
                )
            )
        st.session_state.processed = True
        st.success("Inference complete!")

    # After processing, show download button
    if st.session_state.get("processed", False):
        def cleanup():
            for p in (st.session_state.source_path, st.session_state.target_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            for key in ("uploaded_name", "processed", "source_path", "target_path"):
                st.session_state.pop(key, None)

        with open(target, "rb") as vid:
            st.download_button(
                "Download annotated video",
                data=vid,
                file_name=os.path.basename(target),
                mime="video/mp4",
                on_click=cleanup
            )
else:
    st.info("Awaiting video upload…")
