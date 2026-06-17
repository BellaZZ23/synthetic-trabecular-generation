"""
Quantum-AI micro-CT bone pipeline — live demo
=============================================
Self-contained, single-file Streamlit app. No HPC, no FE solver, no
external data, no special hardware. Dependencies: streamlit, numpy,
scipy, matplotlib.

A volume is auto-generated on load, so every tab is populated the moment
the app opens — nothing to click before presenting.

Narrative (≈5 min) — maps onto the QIC pipeline:
  ① Generate         — synthetic trabecular bone via zero-crossing
                       Gaussian random field (smoothed noise → elastic
                       warp → |score|>τ → binary-search τ to BV/TV →
                       grayscale render). Ground-truth factory.
  ② Segment & validate — algorithmic masking (Otsu) recovers a mask from
                       the grayscale; validated against the KNOWN mask
                       (Dice/IoU/accuracy + Dice-vs-threshold curve).
  ③ Curate & align   — build a deformed sub-volume, recover the rigid
                       offset by phase cross-correlation, re-align.
  ④ Compare          — similarity via a PLUGGABLE backend. Classical NCC
                       is active; the quantum kernel drops into the slot.
  ⑤ Quantum & hybrid — the hybrid workflow (sub-volume → UMAP → quantum
                       feature map → quantum kernel) and the extension
                       points that make the platform quantum-ready.

Run locally:   streamlit run streamlit_app.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import (
    gaussian_filter, distance_transform_edt, map_coordinates, label
)

st.set_page_config(page_title="Quantum-AI bone pipeline — demo",
                   page_icon="🦴", layout="wide")

C_BLUE = "#378ADD"
C_ORANGE = "#E85D3A"
C_GREEN = "#1D9E75"
C_PURPLE = "#7F77DD"
C_BONE = "#C8BFA9"


# ══════════════════════════════════════════════════════════════
# NUMERICAL CORE  (pure, fast, deterministic — verified headless)
# ══════════════════════════════════════════════════════════════

def _norm(a):
    return (a - a.min()) / (np.ptp(a) + 1e-9)


def _smoothed_field(shape, sigma, seed):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal(shape), sigma=sigma)


def _elastic_warp(field, amp, sigma, seed):
    if amp <= 0:
        return field
    rng = np.random.default_rng(seed + 999)
    nz, ny, nx = field.shape
    dz = gaussian_filter(rng.standard_normal(field.shape), sigma) * amp
    dy = gaussian_filter(rng.standard_normal(field.shape), sigma) * amp
    dx = gaussian_filter(rng.standard_normal(field.shape), sigma) * amp
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    return map_coordinates(field, coords, order=1, mode="reflect")


def _boneness(shape, sigma, plate_w, rod_w, warp_amp, warp_sigma, seed):
    f_plate = _smoothed_field(shape, sigma, seed)
    f_plate = _elastic_warp(f_plate, warp_amp, warp_sigma, seed)
    plate_score = -np.abs(f_plate)
    f_rod = _smoothed_field(shape, sigma * 1.7, seed + 7)
    rod_score = f_rod
    return plate_w * _norm(plate_score) + rod_w * _norm(rod_score)


def _threshold_to_bvtv(score, target_bvtv, iters=40):
    lo, hi = float(score.min()), float(score.max())
    tau = 0.5 * (lo + hi)
    for _ in range(iters):
        tau = 0.5 * (lo + hi)
        if float((score > tau).mean()) > target_bvtv:
            lo = tau
        else:
            hi = tau
    return (score > tau).astype(np.uint8), tau


def _make_grayscale(mask, bone_mean, marrow_mean, blur, noise_sd, seed):
    rng = np.random.default_rng(seed + 3)
    dt = distance_transform_edt(mask)
    depth = dt / dt.max() if dt.max() > 0 else dt
    img = np.where(mask > 0,
                   marrow_mean + (bone_mean - marrow_mean) * (0.55 + 0.45 * depth),
                   float(marrow_mean))
    img = gaussian_filter(img, blur) + rng.normal(0, noise_sd, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _quick_morph(mask, voxel_um):
    bvtv = float(mask.mean())
    dt_b = distance_transform_edt(mask)
    inside = dt_b[mask > 0]
    tbth = float(2 * np.median(inside) * voxel_um) if inside.size else 0.0
    dt_m = distance_transform_edt(1 - mask)
    outside = dt_m[mask == 0]
    tbsp = float(2 * np.median(outside) * voxel_um) if outside.size else 0.0
    tbn = bvtv / (tbth / 1000.0) if tbth > 0 else 0.0
    lab, n = label(mask)
    lcc = (np.bincount(lab.ravel())[1:].max() / mask.sum()) if n > 0 else 0.0
    return dict(BVTV=bvtv, TbTh_um=tbth, TbSp_um=tbsp,
                TbN_per_mm=tbn, LCC=float(lcc), n_components=int(n))


# ── Segmentation / algorithmic masking ───────────────────────────────
def otsu_threshold(img):
    """Otsu's method — maximises between-class variance over a 256-bin
    grayscale histogram. Pure numpy, no scikit-image."""
    hist = np.histogram(img.ravel(), bins=256, range=(0, 256))[0].astype(np.float64)
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, best_var, level = 0.0, 0.0, -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between >= best_var:
            best_var = between
            level = t
    return int(level)


def seg_metrics(pred, true):
    pred = pred.astype(bool); true = true.astype(bool)
    tp = int(np.sum(pred & true)); fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true)); tn = int(np.sum(~pred & ~true))
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    acc = (tp + tn) / pred.size
    return dict(dice=dice, iou=iou, acc=acc, tp=tp, fp=fp, fn=fn, tn=tn)


def agreement_rgb(pred2d, true2d):
    """RGB confusion image: TP green, FP red (over-segment), FN blue (missed)."""
    h, w = pred2d.shape
    rgb = np.zeros((h, w, 3))
    pred2d = pred2d.astype(bool); true2d = true2d.astype(bool)
    rgb[pred2d & true2d] = [0.11, 0.62, 0.46]     # TP green
    rgb[pred2d & ~true2d] = [0.91, 0.36, 0.23]    # FP red
    rgb[~pred2d & true2d] = [0.22, 0.54, 0.87]    # FN blue
    return rgb


# ── Alignment + comparison ───────────────────────────────────────────
def _phase_correction(ref, mov):
    F = np.fft.fftn(ref.astype(np.float32))
    M = np.fft.fftn(mov.astype(np.float32))
    cross = F * np.conj(M)
    cross /= np.abs(cross) + 1e-8
    corr = np.abs(np.fft.ifftn(cross))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    return tuple(int(p if p <= s // 2 else p - s) for p, s in zip(peak, ref.shape))


def _roll(vol, shift):
    return np.roll(vol, shift, axis=(0, 1, 2))


def _make_deformed(gray, shift, warp_amp, warp_sigma, noise_sd, seed):
    d = _roll(gray.astype(np.float32), shift)
    d = _elastic_warp(d, warp_amp, warp_sigma, seed + 11)
    rng = np.random.default_rng(seed + 17)
    d = d + rng.normal(0, noise_sd, d.shape)
    return np.clip(d, 0, 255)


def ncc(a, b):
    """Normalised cross-correlation (global), scale/offset invariant."""
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def ncc_map(ref, mov, patch=8):
    ny, nx = ref.shape
    out = np.full((ny // patch, nx // patch), np.nan)
    for iy in range(out.shape[0]):
        for ix in range(out.shape[1]):
            ys, xs = iy * patch, ix * patch
            out[iy, ix] = ncc(ref[ys:ys + patch, xs:xs + patch],
                              mov[ys:ys + patch, xs:xs + patch])
    return out


SIMILARITY_BACKENDS = {
    "Normalised cross-correlation (NCC) — classical baseline": ncc,
    "Quantum kernel similarity — pluggable slot (not yet active)": None,
}


@st.cache_data(show_spinner=False)
def generate_volume(nz, nxy, voxel_um, sigma, plate_w, rod_w,
                    warp_amp, warp_sigma, target_bvtv,
                    bone_mean, marrow_mean, blur, noise_sd, seed):
    shape = (nz, nxy, nxy)
    t0 = time.time()
    score = _boneness(shape, sigma, plate_w, rod_w, warp_amp, warp_sigma, seed)
    mask, tau = _threshold_to_bvtv(score, target_bvtv)
    gray = _make_grayscale(mask, bone_mean, marrow_mean, blur, noise_sd, seed)
    morph = _quick_morph(mask, voxel_um)
    return dict(mask=mask, gray=gray, morph=morph, tau=tau,
                voxel_um=voxel_um, gen_ms=(time.time() - t0) * 1000)


def _slice_fig(arr, voxel_mm, cmap, title, vmin=None, vmax=None, cbar=False):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ext = [0, arr.shape[1] * voxel_mm, 0, arr.shape[0] * voxel_mm]
    im = ax.imshow(arr.T, cmap=cmap, origin="lower", extent=ext,
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    if cbar:
        plt.colorbar(im, ax=ax, shrink=0.82)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR — global generation controls
# ══════════════════════════════════════════════════════════════

st.sidebar.title("🦴 Pipeline controls")
st.sidebar.caption("A volume is generated automatically on load.")

st.sidebar.header("Volume")
nxy = st.sidebar.select_slider("XY size (voxels)", [32, 40, 48, 64], value=48)
nz = st.sidebar.select_slider("Z slices", [24, 32, 40, 48], value=40)
voxel_um = st.sidebar.number_input("Voxel size (µm)", value=39.0, step=1.0)

st.sidebar.header("Morphometric target")
target_bvtv = st.sidebar.slider("Target BV/TV", 0.10, 0.45, 0.30, 0.01)

st.sidebar.header("Architecture")
sigma = st.sidebar.slider("Base sigma (plate scale)", 1.0, 4.0, 2.2, 0.1)
plate_w = st.sidebar.slider("Plate weight", 0.0, 1.0, 0.7, 0.05)
rod_w = st.sidebar.slider("Rod weight", 0.0, 1.0, 0.3, 0.05)
warp_amp = st.sidebar.slider("Elastic warp amplitude", 0.0, 3.0, 2.0, 0.1)
warp_sigma = st.sidebar.slider("Warp correlation length", 4.0, 16.0, 10.0, 0.5)

st.sidebar.header("Grayscale render")
bone_mean = st.sidebar.slider("Bone intensity", 50, 200, 90, 5)
marrow_mean = st.sidebar.slider("Marrow intensity", 5, 50, 15, 1)
blur = st.sidebar.slider("Blur sigma", 0.2, 2.0, 0.8, 0.1)
noise_sd = st.sidebar.slider("Noise SD", 0.0, 6.0, 2.0, 0.5)

st.sidebar.header("Reproducibility")
seed = st.sidebar.number_input("Random seed", value=100, step=1)

gen_clicked = st.sidebar.button("⚙️ Regenerate volume", type="primary",
                                use_container_width=True)

# Auto-preload on first load; regenerate on demand.
if gen_clicked or "vol" not in st.session_state:
    st.session_state["vol"] = generate_volume(
        nz, nxy, voxel_um, sigma, plate_w, rod_w, warp_amp, warp_sigma,
        target_bvtv, bone_mean, marrow_mean, blur, noise_sd, int(seed))


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.title("Quantum-AI synergy for micro-CT bone imaging")
st.caption(
    "Classical backbone of the imaging pipeline, with the quantum "
    "comparison step built as a pluggable slot. "
    "Generate → Segment → Align → Compare → Quantum & hybrid."
)

vol = st.session_state["vol"]
mask, gray, morph = vol["mask"], vol["gray"], vol["morph"]
voxel_mm = vol["voxel_um"] / 1000.0
nz_v, ny_v, nx_v = mask.shape

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Generate", "② Segment & validate", "③ Curate & align",
    "④ Compare", "⑤ Quantum & hybrid",
])


# ──────────────────────────────────────────────────────────────
# ① GENERATE
# ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Synthetic trabecular bone — zero-crossing Gaussian random field")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BV/TV", f"{morph['BVTV']:.3f}", f"target {target_bvtv:.2f}")
    m2.metric("Tb.Th", f"{morph['TbTh_um']:.0f} µm")
    m3.metric("Tb.N", f"{morph['TbN_per_mm']:.2f} /mm")
    m4.metric("LCC", f"{morph['LCC']:.3f}",
              "connected" if morph["LCC"] > 0.95 else "fragmented")
    m5.metric("Generate", f"{vol['gen_ms']:.0f} ms")

    z = st.slider("Z-slice", 0, nz_v - 1, nz_v // 2, key="gen_z")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.pyplot(_slice_fig(mask[z], voxel_mm, "gray",
                             f"Binary mask (z={z})")); plt.close()
    with g2:
        st.pyplot(_slice_fig(gray[z], voxel_mm, "gray",
                             f"Synthetic µCT (z={z})", 0, 255)); plt.close()
    with g3:
        st.pyplot(_slice_fig(gray.max(axis=0), voxel_mm, "gray",
                             "Max-intensity projection", 0, 255)); plt.close()

    st.caption(
        f"Plate/rod weight {plate_w:.1f}/{rod_w:.1f} · {voxel_um:.0f} µm voxels · "
        f"τ binary-searched to BV/TV={morph['BVTV']:.3f} (τ={vol['tau']:.3f}). "
        "Tb.Th/Tb.N are fast distance-transform estimates for live display."
    )


# ──────────────────────────────────────────────────────────────
# ② SEGMENT & VALIDATE  (Otsu + ground-truth validation)
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Segmentation & algorithmic masking — validated against ground truth")
    st.write(
        "Because the volume is synthetic, the **true** bone mask is known. "
        "We recover a mask from the grayscale µCT using an algorithmic "
        "threshold and score it against that ground truth — the kind of "
        "validation that is impossible on real scans alone."
    )

    s1, s2 = st.columns([1, 1])
    with s1:
        method = st.radio("Masking method",
                          ["Otsu (automatic)", "Manual threshold"],
                          key="seg_method", horizontal=True)
    otsu_t = otsu_threshold(gray)
    with s2:
        if method == "Manual threshold":
            thr = st.slider("Threshold", 0, 255, int(otsu_t), 1, key="seg_thr")
        else:
            thr = otsu_t
            st.metric("Otsu threshold", f"{otsu_t}")

    pred = (gray >= thr).astype(np.uint8)
    met = seg_metrics(pred, mask)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Dice", f"{met['dice']:.3f}")
    v2.metric("IoU", f"{met['iou']:.3f}")
    v3.metric("Voxel accuracy", f"{met['acc']:.3f}")
    v4.metric("BV/TV (pred vs true)",
              f"{pred.mean():.3f}", f"{pred.mean() - mask.mean():+.3f}")

    sz = st.slider("Z-slice", 0, nz_v - 1, nz_v // 2, key="seg_z")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(_slice_fig(gray[sz], voxel_mm, "gray",
                             f"Grayscale µCT (z={sz})", 0, 255)); plt.close()
    with c2:
        st.pyplot(_slice_fig(pred[sz], voxel_mm, "gray",
                             f"Recovered mask ({method.split()[0]})")); plt.close()
    with c3:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ext = [0, nx_v * voxel_mm, 0, ny_v * voxel_mm]
        ax.imshow(np.transpose(agreement_rgb(pred[sz].T, mask[sz].T), (1, 0, 2)),
                  origin="lower", extent=ext)
        ax.set_title("Agreement (TP green / FP red / FN blue)", fontsize=9)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("##### Validation: Dice vs threshold")
    st.caption("Otsu lands near the optimum without being told the answer — "
               "the curve is the evidence.")
    ts = np.arange(0, 256, 4)
    dices = [seg_metrics((gray >= t).astype(np.uint8), mask)["dice"] for t in ts]
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.plot(ts, dices, color=C_BLUE, lw=2)
    ax.axvline(otsu_t, color=C_ORANGE, ls="--", lw=2, label=f"Otsu = {otsu_t}")
    if method == "Manual threshold":
        ax.axvline(thr, color=C_GREEN, ls=":", lw=2, label=f"manual = {thr}")
    ax.set_xlabel("threshold"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1)
    ax.legend(loc="lower center"); ax.set_xlim(0, 255)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────
# ③ CURATE & ALIGN
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Curate & align — recover the rigid offset between two states")
    st.write(
        "A second 'deformed' sub-volume is created with a **known** rigid "
        "offset plus a mild elastic warp and noise — standing in for a "
        "second DVC load step. Phase cross-correlation recovers the offset; "
        "the volume is re-aligned before any comparison."
    )

    a1, a2, a3 = st.columns(3)
    sx = a1.slider("Applied shift x", -5, 5, 2, key="sx")
    sy = a2.slider("Applied shift y", -5, 5, 3, key="sy")
    dwarp = a3.slider("Deformation warp", 0.0, 2.0, 0.8, 0.1, key="dwarp")

    applied = (0, int(sy), int(sx))
    deformed = _make_deformed(gray, applied, dwarp, 8.0, noise_sd, int(seed))
    correction = _phase_correction(gray, deformed)
    aligned = _roll(deformed, correction)
    est_offset = tuple(-c for c in correction)

    # share with the Compare tab (runs later in the script)
    st.session_state["aligned"] = aligned
    st.session_state["compare_ref"] = gray

    e1, e2 = st.columns(2)
    e1.metric("Applied offset (x, y)", f"({applied[2]}, {applied[1]})")
    e2.metric("Recovered offset (x, y)", f"({est_offset[2]}, {est_offset[1]})",
              "✓ match" if est_offset == applied else "residual")

    cz = st.slider("Z-slice", 0, nz_v - 1, nz_v // 2, key="align_z")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.pyplot(_slice_fig(gray[cz], voxel_mm, "gray",
                             "Reference", 0, 255)); plt.close()
    with p2:
        st.pyplot(_slice_fig(deformed[cz], voxel_mm, "gray",
                             "Deformed (before align)", 0, 255)); plt.close()
    with p3:
        st.pyplot(_slice_fig(aligned[cz], voxel_mm, "gray",
                             "Deformed (after align)", 0, 255)); plt.close()

    ref2d, ali2d = gray[cz].T, aligned[cz].T
    p = 8
    ii = np.arange(ref2d.shape[0])[:, None] // p
    jj = np.arange(ref2d.shape[1])[None, :] // p
    cb = np.where(((ii + jj) % 2) == 1, ali2d, ref2d)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ext = [0, nx_v * voxel_mm, 0, ny_v * voxel_mm]
    ax.imshow(cb, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=255)
    ax.set_title("Checkerboard QA — features should line up across tiles",
                 fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────
# ④ COMPARE  (pluggable backend)
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Compare — similarity via a pluggable backend")
    st.write(
        "The comparison step is **backend-agnostic**. The classical NCC "
        "baseline is active now; the quantum kernel implements the same "
        "interface and drops into this exact slot — nothing else in the "
        "pipeline changes."
    )

    backend_name = st.selectbox("Similarity backend",
                                list(SIMILARITY_BACKENDS.keys()))
    backend = SIMILARITY_BACKENDS[backend_name]
    ref = st.session_state.get("compare_ref", gray)
    aligned = st.session_state.get("aligned")

    if backend is None:
        st.warning(
            "🔒 **Quantum kernel slot — reserved, not yet active.**\n\n"
            "This backend is registered but intentionally inert in the demo. "
            "See tab **⑤ Quantum & hybrid** for the interface it implements."
        )
    elif aligned is None:
        st.info("Open **③ Curate & align** once to create an aligned pair.")
    else:
        cz = nz_v // 2
        global_r = backend(ref, aligned)
        sim = ncc_map(ref[cz].T, aligned[cz].T, patch=8)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Global similarity (NCC)", f"{global_r:.4f}")
            st.metric("Patch grid", f"{sim.shape[1]}×{sim.shape[0]}")
            st.caption("1.0 = identical · 0 = uncorrelated. Residual elastic "
                       "warp + noise keep it below 1.")
        with c2:
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            im = ax.imshow(sim, cmap="RdYlGn", origin="lower", vmin=-1, vmax=1)
            ax.set_title("Patch-wise similarity map (reference vs aligned)",
                         fontsize=10)
            ax.set_xlabel("patch x"); ax.set_ylabel("patch y")
            plt.colorbar(im, ax=ax, label="NCC")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.hist(sim[np.isfinite(sim)].ravel(), bins=20, color=C_BLUE,
                alpha=0.85, edgecolor="none")
        ax.set_xlabel("patch NCC"); ax.set_ylabel("count")
        ax.set_title("Distribution of patch similarities", fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────
# ⑤ QUANTUM & HYBRID
# ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Quantum & hybrid workflow — where the kernel plugs in")
    st.markdown(
        "The comparison backend is the only place the quantum step enters. "
        "A sub-volume is too high-dimensional for a quantum feature map "
        "directly, so the workflow is **hybrid**: classical dimensionality "
        "reduction compresses each sub-volume, then a quantum feature map "
        "and kernel measure similarity."
    )

    fig, ax = plt.subplots(figsize=(9.2, 1.9))
    ax.axis("off")
    stages = [("Sub-volume", C_BONE, False), ("UMAP\nreduce", C_BLUE, False),
              ("Quantum\nfeature map", C_PURPLE, True),
              ("Quantum\nkernel", C_PURPLE, True),
              ("Similarity", C_GREEN, False)]
    for i, (lab, col, q) in enumerate(stages):
        x = i * 1.95
        ax.add_patch(plt.Rectangle((x, 0), 1.6, 1.0, facecolor=col,
                     edgecolor="black", alpha=0.9,
                     linestyle="--" if q else "-", linewidth=2))
        ax.text(x + 0.8, 0.5, lab, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white" if col != C_BONE else "black")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.95, 0.5), xytext=(x + 1.6, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(2.0 + 0.8, 1.28, "quantum (pluggable)", ha="center", fontsize=8,
            style="italic", color=C_PURPLE)
    ax.set_xlim(-0.2, 10.0); ax.set_ylim(-0.2, 1.65)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**The published design rule**")
    st.info(
        "For trabecular-bone classification, the dimensionality-reduction "
        "method decides whether the quantum kernel stays competitive: "
        "**UMAP preserves quantum–classical parity**, while linear methods "
        "(PCA / random projection / PLS) lose ~9–12 points. So the reducer "
        "in the box above is not incidental — it is the deciding factor. "
        "Long-term, this comparison becomes Quantum Image Correlation (QIC), "
        "a successor to digital volume correlation (DVC)."
    )

    st.markdown("**Extension points — the platform is quantum-ready**")
    st.code(
        '# 1. Register a feature reducer (classical pre-processing)\n'
        'FEATURE_REDUCERS = {\n'
        '    "UMAP":  umap_reduce,      # quantum-classical parity\n'
        '    "PCA":   pca_reduce,       # baseline\n'
        '}\n\n'
        '# 2. Register a similarity backend (classical or quantum)\n'
        'SIMILARITY_BACKENDS = {\n'
        '    "NCC (classical)": ncc,\n'
        '    "Quantum kernel":  quantum_similarity,   # <- drops in here\n'
        '}\n\n'
        'def quantum_similarity(ref_patch, def_patch):\n'
        '    """Scalar in [-1, 1]. Fidelity of a quantum feature map,\n'
        '    e.g. |<phi(ref)|phi(def)>|^2 from a Qiskit quantum kernel."""\n'
        '    ...',
        language="python",
    )

    st.markdown("**Future workflows this scaffolding supports**")
    st.markdown(
        "- Quantum-kernel SVM classification of sub-volumes (the published result)\n"
        "- Full QIC displacement/strain fields via per-sub-volume quantum similarity\n"
        "- Hybrid quantum-classical FE coupling (mechanically-aware generation)\n"
        "- Self-supervised feature extraction (e.g. DINO) feeding the reducer slot"
    )

    st.success(
        "Status: classical backbone complete · segmentation validated against "
        "ground truth · quantum slot defined and registered — kernel drops in next."
    )