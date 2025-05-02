import streamlit as st
import cv2, os, tempfile

st.set_page_config(page_title='Frame Analyzer')
st.title("📹 Frame Analyzer")

uploader = st.file_uploader('Upload video', type=['mp4','avi','mov'])
if not uploader:
    st.write('Need to upload file to continue')
else:
    # ─── one-time init ──────────────────────────────────────────────
    if "cap" not in st.session_state or st.session_state.uploaded_name != uploader.name:
        # write upload to temp
        ext = os.path.splitext(uploader.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(uploader.read()); tmp.flush(); tmp.close()
        st.session_state.uploaded_name = uploader.name
        st.session_state.cap = cv2.VideoCapture(tmp.name)
        # store frame_count too
        st.session_state.frame_count = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.frame_idx   = 0

    cap         = st.session_state.cap
    frame_count = st.session_state.frame_count

    # ─── always show these on every run ────────────────────────────
    st.write(f"Total frames: {frame_count}")

    # frame picker bound to session_state
    st.number_input(
        "Frame#",
        min_value=0,
        max_value=frame_count-1,
        value=st.session_state.frame_idx,
        step=1,
        format="%d",
        key="frame_idx",
    )
    frame_idx = st.session_state.frame_idx

    # seek & display
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        st.error(f"Couldn’t read frame {frame_idx}")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, use_column_width=True)



