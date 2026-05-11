"""
Quantum Bone Pipeline — Streamlit Dashboard
============================================
Entry point. Run with:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Quantum Bone Pipeline",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Quantum bone pipeline")
st.sidebar.caption("v15.3 zero-crossing generator + micro-FE")

st.markdown("""
# Quantum bone pipeline dashboard

This dashboard wraps the published quantum kernel SVM pipeline for
trabecular bone classification, extended with mechanical awareness
via micro-FE coupling.

### Pages

- **Bone generator** — generate synthetic trabecular bone volumes with
  controllable morphometric parameters (BV/TV, Tb.Th, plate/rod weight)

- **FE solver** — apply compression, tension, or torque loads to generated
  volumes, visualise displacement and strain fields, validate against
  Voigt bounds

Use the sidebar to navigate between pages.

---
*Isabella Florez — University of Greenwich, School of Engineering*
""")
