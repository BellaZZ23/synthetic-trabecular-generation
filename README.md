# Quantum-AI micro-CT bone pipeline — live demo

Self-contained Streamlit app for the 18 June research presentation. No
HPC, no FE solver, no external data, no special hardware. Each step runs
in well under a second on a 48×48×40 volume, and a volume is generated
automatically on load, so it is safe to run live.

**Flow:** ① Generate → ② Segment & validate → ③ Curate & align → ④ Compare → ⑤ Quantum & hybrid

This is the classical backbone of the Quantum Image Correlation (QIC)
pipeline, with the comparison step built as a pluggable slot the quantum
kernel drops into.

## Files
- `streamlit_app.py` — the whole app (single file)
- `requirements.txt` — `streamlit`, `numpy`, `scipy`, `matplotlib`
- `METHODS.md` — segmentation, masking, validation, and quantum/hybrid notes

## Run locally (recommended for the talk)
From the folder containing `streamlit_app.py`, in your venv:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

It opens at http://localhost:8501. The first tab is already populated —
walk left to right through the five tabs. Running locally needs no
internet during the talk, which is the safest option.

## Data is pre-loaded
There is nothing to upload and no blank screen on stage. On load the app
deterministically generates a synthetic volume (fixed seed), so every tab
is populated immediately. The sidebar **Regenerate volume** button makes a
fresh one; everything downstream (segmentation, alignment, comparison)
recomputes automatically.

If you later want to demo a *real* scan instead of synthetic data, commit
a small `sample_volume.npy` to the repo and add a loader — but for a live
demo the deterministic synthetic volume is the most reliable choice.

## Publish it as an app (Streamlit Community Cloud)
Deployment is free and the flow is stable:

1. Put `streamlit_app.py` and `requirements.txt` in a **public** GitHub
   repo (add them to `BellaZZ23/quantum-bone-classification`, or a small
   new repo). Commit and push.
2. Go to https://share.streamlit.io and sign in with GitHub (accept the
   permissions on first use).
3. Click **Create app** (upper-right) → **"Yup, I have an app."**
4. Fill in the repository, branch, and file path (`streamlit_app.py`).
   Optionally set a custom subdomain.
5. Click **Deploy**. After a few minutes you get a public
   `https://<your-subdomain>.streamlit.app` URL.

The app is tiny, so it stays well within the free tier's memory limit,
and every `git push` redeploys it automatically. Community Cloud supports
all current Python versions. Treat the hosted link as a backup — run
locally for the live demo itself.

## Demo tips
- Default volume is 48×48×40. Pushing XY to 64 is still fast, but leave
  margin when presenting.
- The sidebar is the control panel; the five numbered tabs are the
  narrative. Move left to right.
- Tab ② is the segmentation + validation story (Otsu vs the known mask).
- Tab ⑤ shows the hybrid workflow and exactly where the quantum kernel
  plugs in (the `SIMILARITY_BACKENDS` registry in `streamlit_app.py`).