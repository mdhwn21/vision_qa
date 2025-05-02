import streamlit as st
import cv2
import os
import tempfile

st.set_page_config(page_title='Frame Analyzer')
st.title("📹 Frame Analyzer")

# 1) Upload video
uploader = st.file_uploader('Upload video', type=['mp4','avi','mov'])
if not uploader:
    st.write('Need to upload file to continue')
else:
    # ─── one-time init ──────────────────────────────────────────────
    if "cap" not in st.session_state or st.session_state.uploaded_name != uploader.name:
        # Write upload to temp file
        ext = os.path.splitext(uploader.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(uploader.read()); tmp.flush(); tmp.close()

        # Store in session
        st.session_state.uploaded_name = uploader.name
        st.session_state.video_path    = tmp.name
        st.session_state.cap           = cv2.VideoCapture(tmp.name)
        st.session_state.frame_count   = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.frame_idx     = 0

    cap = st.session_state.cap
    frame_count = st.session_state.frame_count

    st.write(f"Total frames: {frame_count}")

    # ─── frame selector ─────────────────────────────────────────────
    st.number_input(
        "Frame#",
        min_value=0,
        max_value=frame_count - 1,
        value=st.session_state.frame_idx,
        step=1,
        format="%d",
        key="frame_idx"
    )
    frame_idx = st.session_state.frame_idx

    # ─── display selected frame ─────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        st.error(f"Couldn’t read frame {frame_idx}")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, use_column_width=True)

    # ─── export frames ──────────────────────────────────────────────
    st.subheader("Export Frames")
    dest_dir = st.text_input(
        "Destination directory (relative or absolute)",
        value="frames",
        key="export_dir"
    )
    export_spec = st.text_input(
        "Frames to export (blank = all, number = single, range e.g. 1-10):",
        key="export_spec"
    )

    if st.button("Export"):
        abs_dest = os.path.abspath(dest_dir)
        os.makedirs(abs_dest, exist_ok=True)
        cap_export = cv2.VideoCapture(st.session_state.video_path)
        total = frame_count
        width = len(str(total))
        exported = 0

        spec = export_spec.strip()
        # Determine range
        if spec == "":
            start, end = 0, total - 1
        else:
            if '-' in spec:
                parts = spec.split('-', 1)
                try:
                    start = int(parts[0].strip())
                    end   = int(parts[1].strip())
                except ValueError:
                    st.error("Invalid range format; use 'start-end'.")
                    cap_export.release()
                    st.stop()
            else:
                try:
                    idx = int(spec)
                    start, end = idx, idx
                except ValueError:
                    st.error("Invalid frame number.")
                    cap_export.release()
                    st.stop()

        # Clamp and validate
        start = max(0, min(start, total - 1))
        end   = max(0, min(end, total - 1))
        if start > end:
            st.error("Start frame must be ≤ end frame.")
            cap_export.release()
            st.stop()

        # Export frames
        for i in range(start, end + 1):
            cap_export.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frm = cap_export.read()
            if not ret:
                break
            # Save BGR directly (no color conversion)
            filename = os.path.join(abs_dest, f"img_{i:0{width}d}.png")
            cv2.imwrite(filename, frm)
            exported += 1

        cap_export.release()
        st.success(f"Exported {exported} frames to '{abs_dest}'")


