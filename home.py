import streamlit as st

st.set_page_config(
    page_title="Vision-QA",
    page_icon="🏓",
    layout="wide"
)

st.title("🏓 Vision-QA Dashboard")
st.markdown("""
Use the sidebar (or navigation menu) to switch between:

- **Running The Model** — step through video frames  
- **Frame Analyzer** — find court lines & corners  
""")
