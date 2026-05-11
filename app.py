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

- **Data loader** — load real micro-CT volumes to extract morphometric
  parameters for synthetic generation, or enter metrics manually from
  published data

- **Bone generator** — generate synthetic trabecular bone volumes with
  controllable morphometric parameters (BV/TV, Tb.Th, plate/rod weight)

- **FE solver** — apply compression, tension, or torque loads to generated
  volumes, visualise displacement and strain fields, validate against
  Voigt bounds

- **Pipeline** — end-to-end workflow: generate synthetic volumes with
  grayscale micro-CT and mechanical fields in one click, load real
  mechanical data (DIC strain maps, FE exports), and compare synthetic
  vs real side-by-side

- **3D viewer** — interactive 3D bone model with strain overlay,
  load strain maps from TIFF stacks, run FE and visualise displacement
  and strain fields on the bone surface

Use the sidebar to navigate between pages.

---
*Isabella Florez — University of Greenwich, School of Engineering*
""")