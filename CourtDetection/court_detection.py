import json
import streamlit as st

st.set_page_config(page_title="Court Calibrator (via PolygonZone)", layout="wide")
st.title("🎾 Court Coordinate Calibrator")

st.markdown("""
1. **Click the link below** to open [PolygonZone](https://polygonzone.roboflow.com) in a new tab.  
2. Drag&drop your court image there, draw your polygon, then hit **Copy JSON** (or **Copy Python**) in their toolbar.  
3. Come back here and paste the JSON (or NumPy) string into the box below.
""")

# External-tool link
st.markdown("[👉 Open PolygonZone annotator](https://polygonzone.roboflow.com){:target=\"_blank\"}")

# Paste back in
coords_txt = st.text_area(
    "Paste the PolygonZone JSON output here",
    placeholder='e.g. [[215,566],[509,64],[798,67],[1171,564]]'
)
if coords_txt:
    try:
        coords = json.loads(coords_txt)
        st.success(f"Loaded {len(coords)} points.")
        st.write(coords)  # or more preview logic…
    except Exception:
        st.error("Could not parse that input—make sure it’s valid JSON from PolygonZone.")
