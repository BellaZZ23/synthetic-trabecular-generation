# -*- coding: utf-8 -*-
"""
Page 8: Foam Materials
=======================
Micro-CT analysis and synthetic generation for open-cell and
closed-cell foam-like porous materials.

Workflow:
  1. Generate a synthetic foam volume (GRF open-cell or sphere-pack closed-cell)
  2. Measure foam morphometrics (porosity, strut diameter, cell size, anisotropy)
  3. Predict mechanical properties via the Gibson-Ashby model
  4. Load real foam micro-CT data and compare predicted vs measured
  5. Push morphometrics to the pipeline as material targets

Connection to the rest of the tool:
  - Foam morphometrics are stored in session_state["foam_morphometrics"]
  - The FE solver accepts a material preset key "foam" that uses
    the Gibson-Ashby modulus and density instead of bone tissue values
  - The QKSVM page can classify foam specimens by density class
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from scipy.ndimage import (
    gaussian_filter, label as ndi_label,
    distance_transform_edt, binary_erosion, binary_dilation,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ui_style import inject_css, page_header, section_label

st.set_page_config(page_title="Foam materials", page_icon="🧽", layout="wide")
inject_css()
from ui_style import qic_pipeline_sidebar
qic_pipeline_sidebar()
page_header(
    title="Foam materials",
    subtitle="Synthetic generation · morphometrics · Gibson-Ashby mechanics · micro-CT import",
    label="Extension · Foam",
    color="#E85D3A",
)

C_TEAL   = "#1D9E75"
C_BLUE   = "#378ADD"
C_CORAL  = "#E85D3A"
C_PURPLE = "#7F77DD"

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def generate_open_cell_foam(
    size: int = 64,
    porosity: float = 0.85,
    strut_scale: float = 3.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Open-cell foam via Gaussian random field zero-crossing.
    Same approach as trabecular bone but calibrated to foam morphology:
    higher porosity (0.7–0.97), finer strut scale.
    Returns binary array: 1 = solid strut, 0 = pore space.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size, size))
    smooth = gaussian_filter(noise, sigma=strut_scale)
    solid_frac = 1.0 - porosity
    thresh = np.percentile(smooth, (1.0 - solid_frac) * 100)
    return (smooth >= thresh).astype(np.uint8)


def generate_closed_cell_foam(
    size: int = 64,
    porosity: float = 0.75,
    cell_radius_vox: float = 6.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Closed-cell foam via random sphere packing with thin walls.
    Spheres = pore cells; voxels outside all spheres = solid wall.
    """
    rng = np.random.default_rng(seed)
    vol = np.zeros((size, size, size), dtype=np.float32)
    n_spheres = max(1, int(round(size**3 * (1 - porosity) / (4/3 * np.pi * cell_radius_vox**3) * 3)))
    n_spheres = min(n_spheres, 500)
    centres = rng.uniform(cell_radius_vox, size - cell_radius_vox, (n_spheres, 3))
    zz, yy, xx = np.ogrid[:size, :size, :size]
    for c in centres:
        d2 = (zz - c[0])**2 + (yy - c[1])**2 + (xx - c[2])**2
        vol += np.exp(-d2 / (2 * (cell_radius_vox * 0.6)**2))
    # Threshold: pore inside spheres, solid in walls
    thresh = np.percentile(vol[vol > 0], 40) if vol.max() > 0 else 0.1
    solid = (vol < thresh).astype(np.uint8)
    return solid


def measure_foam_morphometrics(
    solid: np.ndarray, voxel_um: float = 39.0
) -> dict:
    """
    Measure key foam morphometrics from a binary solid mask.

    Returns
    -------
    dict with keys:
        porosity        : void fraction (0–1)
        solid_fraction  : 1 − porosity
        strut_dia_um    : mean strut/wall thickness via local thickness (µm)
        cell_size_um    : mean pore cell diameter (µm)
        n_cells         : approximate number of pore cells
        sa_density_mm2mm3 : surface area density (mm²/mm³)
        connectivity_density : connected solid component density (/mm³)
    """
    pore = 1 - solid
    solid_frac = float(solid.mean())
    porosity   = 1.0 - solid_frac

    # Strut/wall thickness: mean local thickness from distance transform on SOLID
    dt_solid = distance_transform_edt(solid) * voxel_um
    strut_dia = float(dt_solid[solid.astype(bool)].mean()) * 2.0  # diameter

    # Cell size: mean local thickness in PORE space
    dt_pore = distance_transform_edt(pore) * voxel_um
    cell_size = float(dt_pore[pore.astype(bool)].mean()) * 2.0

    # Number of pore cells (connected components in pore space)
    _, n_cells = ndi_label(pore)

    # Surface area density (voxel face counting)
    from scipy.ndimage import convolve
    kernel = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                       [[0,1,0],[1,-6,1],[0,1,0]],
                       [[0,0,0],[0,1,0],[0,0,0]]], dtype=np.float32)
    lapl = convolve(solid.astype(np.float32), kernel)
    surface_vox = np.sum(np.abs(lapl) > 0)
    vox_mm = voxel_um / 1000.0
    vol_mm3 = solid.size * vox_mm**3
    sa_dens = surface_vox * vox_mm**2 / vol_mm3

    # Connectivity density: largest connected component fraction
    lbl, n_comp = ndi_label(solid)
    if n_comp > 0:
        sizes = np.bincount(lbl.ravel())[1:]
        conn_frac = sizes.max() / solid.sum() if solid.sum() > 0 else 0.0
    else:
        conn_frac = 0.0

    return {
        "porosity":           round(porosity, 4),
        "solid_fraction":     round(solid_frac, 4),
        "strut_dia_um":       round(strut_dia, 1),
        "cell_size_um":       round(cell_size, 1),
        "n_cells":            int(n_cells),
        "sa_density_mm2mm3":  round(sa_dens, 3),
        "largest_cc_fraction": round(conn_frac, 3),
    }


def gibson_ashby_modulus(
    solid_fraction: float,
    E_solid_MPa: float = 3500.0,
    foam_type: str = "open",
) -> float:
    """
    Gibson-Ashby scaling law for foam Young's modulus.

    Open-cell:   E* / E_s ≈ C₁ (ρ*/ρ_s)²
    Closed-cell: E* / E_s ≈ C₁ φ² (ρ*/ρ_s)² + C₂ (1−φ)(ρ*/ρ_s)
    where C₁ = 1 (open), φ = solid face fraction ≈ 0.6 for typical foams.
    """
    rho_rel = solid_fraction
    if foam_type == "open":
        ratio = rho_rel ** 2
    else:
        phi = 0.6
        ratio = phi**2 * rho_rel**2 + (1 - phi) * rho_rel
    return E_solid_MPa * ratio


def gibson_ashby_strength(
    solid_fraction: float,
    sigma_solid_MPa: float = 80.0,
    foam_type: str = "open",
) -> float:
    """Gibson-Ashby compressive strength scaling."""
    rho_rel = solid_fraction
    if foam_type == "open":
        return sigma_solid_MPa * 0.3 * rho_rel**1.5
    else:
        phi = 0.6
        return sigma_solid_MPa * 0.3 * (phi * rho_rel**1.5 + (1 - phi) * rho_rel)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.header("Foam type")
foam_type = st.sidebar.radio("Cell type", ["Open-cell", "Closed-cell"],
    help="Open-cell: interconnected pores (reticulated). "
         "Closed-cell: sealed cells with thin walls (Kelvin foam).")
foam_key = "open" if foam_type == "Open-cell" else "closed"

st.sidebar.divider()
st.sidebar.header("Generation parameters")
vol_size = st.sidebar.slider("Volume size (voxels³)", 32, 128, 64, 16, key="foam_size")
voxel_um = st.sidebar.number_input("Voxel size (µm)", value=39.0, step=1.0, key="foam_vox")
seed     = st.sidebar.number_input("Seed", value=42, step=1, key="foam_seed")

if foam_key == "open":
    porosity    = st.sidebar.slider("Target porosity", 0.60, 0.97,
                    st.session_state.get("foam_porosity", 0.85), 0.01, key="foam_porosity")
    strut_scale = st.sidebar.slider("Strut scale (voxels σ)", 1.5, 8.0,
                    st.session_state.get("foam_strut", 3.5), 0.5, key="foam_strut")
else:
    porosity      = st.sidebar.slider("Target porosity", 0.50, 0.90,
                      st.session_state.get("foam_porosity", 0.75), 0.01, key="foam_porosity")
    cell_r        = st.sidebar.slider("Cell radius (voxels)", 3.0, 12.0,
                      st.session_state.get("foam_cellr", 6.0), 0.5, key="foam_cellr")

st.sidebar.divider()
st.sidebar.header("Base material")
MATERIAL_PRESETS = {
    "Polymer (PLA)":    {"E_MPa": 3500, "sigma_MPa": 80,  "rho_gcm3": 1.24},
    "Aluminium foam":   {"E_MPa": 70000,"sigma_MPa": 270, "rho_gcm3": 2.70},
    "Polyurethane foam":{"E_MPa": 40,   "sigma_MPa": 1.5, "rho_gcm3": 0.03},
    "Bone mineral":     {"E_MPa": 18000,"sigma_MPa": 200, "rho_gcm3": 3.16},
    "Custom":           {"E_MPa": 3500, "sigma_MPa": 80,  "rho_gcm3": 1.24},
}
material = st.sidebar.selectbox("Solid material", list(MATERIAL_PRESETS.keys()), key="foam_mat")
mp = MATERIAL_PRESETS[material]
if material == "Custom":
    E_solid   = st.sidebar.number_input("E_solid (MPa)", value=3500.0, step=100.0)
    sig_solid = st.sidebar.number_input("σ_solid (MPa)", value=80.0, step=5.0)
else:
    E_solid   = float(mp["E_MPa"])
    sig_solid = float(mp["sigma_MPa"])

# Preset buttons
st.sidebar.divider()
st.sidebar.caption("Quick presets")
_pc1, _pc2 = st.sidebar.columns(2)
if _pc1.button("🪵 Low density", use_container_width=True, key="foam_preset_low"):
    st.session_state["foam_porosity"] = 0.92
    if foam_key == "open": st.session_state["foam_strut"] = 2.0
    else: st.session_state["foam_cellr"] = 9.0
    st.rerun()
if _pc2.button("🪨 High density", use_container_width=True, key="foam_preset_high"):
    st.session_state["foam_porosity"] = 0.65
    if foam_key == "open": st.session_state["foam_strut"] = 5.0
    else: st.session_state["foam_cellr"] = 4.0
    st.rerun()


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_gen, tab_morph, tab_mech, tab_load, tab_connect = st.tabs([
    "🧽 Generate",
    "📐 Morphometrics",
    "⚙️ Mechanics (Gibson-Ashby)",
    "📂 Load real foam µCT",
    "🔗 Connect to pipeline",
])


# ──────────────────────────────────────────────────────────────
# TAB 1 — GENERATE
# ──────────────────────────────────────────────────────────────
with tab_gen:
    st.markdown(
        f"Generating a **{foam_type.lower()}** foam with porosity "
        f"**{porosity:.0%}** at {vol_size}³ voxels."
    )
    if st.button("▶ Generate foam volume", type="primary", key="btn_gen_foam"):
        with st.spinner("Generating…"):
            if foam_key == "open":
                solid = generate_open_cell_foam(vol_size, porosity, strut_scale, int(seed))
            else:
                solid = generate_closed_cell_foam(vol_size, porosity, cell_r, int(seed))
            m = measure_foam_morphometrics(solid, voxel_um)
            st.session_state["foam_solid"]  = solid
            st.session_state["foam_morph"]  = m
            st.session_state["foam_type"]   = foam_key
            st.session_state["foam_voxel_um"] = voxel_um
            st.session_state["foam_E_MPa"]  = gibson_ashby_modulus(m["solid_fraction"], E_solid, foam_key)
            st.session_state["foam_material"] = material

    if "foam_solid" in st.session_state:
        solid = st.session_state["foam_solid"]
        m     = st.session_state["foam_morph"]
        nz, ny, nx = solid.shape
        vox_mm = voxel_um / 1000.0

        # Metrics strip
        mc = st.columns(5)
        mc[0].metric("Porosity",     f"{m['porosity']:.1%}")
        mc[1].metric("Strut dia",    f"{m['strut_dia_um']:.0f} µm")
        mc[2].metric("Cell size",    f"{m['cell_size_um']:.0f} µm")
        mc[3].metric("Pore cells",   str(m["n_cells"]))
        mc[4].metric("SA density",   f"{m['sa_density_mm2mm3']:.2f} mm²/mm³")

        # Slice gallery
        st.markdown("#### XY · XZ · YZ mid-slices")
        gc1, gc2, gc3 = st.columns(3)
        for col, slc, lbl in [
            (gc1, solid[nz//2, :, :], "XY (mid Z)"),
            (gc2, solid[:, ny//2, :], "XZ (mid Y)"),
            (gc3, solid[:, :, nx//2], "YZ (mid X)"),
        ]:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(slc.T, cmap="bone_r", origin="lower",
                      extent=[0, slc.shape[0]*vox_mm, 0, slc.shape[1]*vox_mm])
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("mm"); ax.set_ylabel("mm")
            col.pyplot(fig); plt.close()

        # Pore size distribution
        with st.expander("Pore-size distribution"):
            dt_pore = distance_transform_edt(1 - solid) * voxel_um
            pore_mask = (1 - solid).astype(bool)
            radii = dt_pore[pore_mask] * 2  # cell diameter proxy
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(radii[radii > 0], bins=40, color=C_CORAL, alpha=0.8, edgecolor="white")
            ax.set_xlabel("Local pore diameter (µm)")
            ax.set_ylabel("Voxel count")
            ax.set_title("Pore-size distribution")
            st.pyplot(fig); plt.close()
    else:
        st.info("Click **Generate foam volume** to start.")


# ──────────────────────────────────────────────────────────────
# TAB 2 — MORPHOMETRICS
# ──────────────────────────────────────────────────────────────
with tab_morph:
    if "foam_morph" not in st.session_state:
        st.info("Generate a foam volume first.")
    else:
        m  = st.session_state["foam_morph"]
        ft = st.session_state.get("foam_type", "open")
        vox = st.session_state.get("foam_voxel_um", 39.0)

        st.markdown("### Foam morphometric report")
        st.markdown(
            f"**Type:** {'Open-cell' if ft=='open' else 'Closed-cell'} · "
            f"**Voxel:** {vox:.0f} µm"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Structural parameters**")
            st.dataframe({
                "Parameter":  ["Porosity", "Solid fraction", "Strut/wall dia (µm)",
                                "Cell size (µm)", "Pore cells"],
                "Value":      [f"{m['porosity']:.4f}", f"{m['solid_fraction']:.4f}",
                               f"{m['strut_dia_um']:.1f}", f"{m['cell_size_um']:.1f}",
                               str(m["n_cells"])],
            }, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("**Quality indicators**")
            cc = m["largest_cc_fraction"]
            sa = m["sa_density_mm2mm3"]
            st.dataframe({
                "Indicator":  ["SA density (mm²/mm³)", "Largest CC fraction",
                                "Strut/cell ratio", "Open-cell character"],
                "Value":      [f"{sa:.3f}", f"{cc:.3f}",
                               f"{m['strut_dia_um']/max(m['cell_size_um'],1):.3f}",
                               "Open" if cc > 0.9 else "Partially closed"],
            }, hide_index=True, use_container_width=True)

        st.markdown(
            "**How to read these:** strut diameter and cell size are computed "
            "from the 3D distance transform — the same method as trabecular "
            "Tb.Th and Tb.Sp — so they are directly comparable to bone morphometrics "
            "when you switch material in the pipeline."
        )

        # Comparison to typical foam ranges
        st.markdown("### Reference ranges (literature)")
        ref_data = {
            "Material":          ["Aluminium foam (Alporas)", "PLA lattice (FDM)",
                                   "Polyurethane RPC", "Cancellous bone", "This foam"],
            "Porosity":          ["0.89–0.93", "0.60–0.85", "0.95–0.98", "0.70–0.90",
                                   f"{m['porosity']:.2f}"],
            "Cell size (µm)":    ["2000–5000", "500–3000", "300–800", "300–700",
                                   f"{m['cell_size_um']:.0f}"],
            "Strut dia (µm)":    ["100–400", "200–800", "30–80", "80–200",
                                   f"{m['strut_dia_um']:.0f}"],
        }
        st.dataframe(ref_data, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 — MECHANICS
# ──────────────────────────────────────────────────────────────
with tab_mech:
    st.markdown("### Gibson-Ashby mechanical property prediction")
    st.markdown(
        "The Gibson-Ashby model relates foam effective modulus and strength "
        "to the solid fraction (relative density) via power-law scaling. "
        "It is the standard first-principles model for cellular solids."
    )
    st.latex(
        r"E^* = C_1 E_s \left(\frac{\rho^*}{\rho_s}\right)^n, \quad "
        r"n = 2 \text{ (open-cell)}, \; n \approx 1.5 \text{ (closed-cell)}"
    )

    if "foam_morph" not in st.session_state:
        st.info("Generate a foam volume first.")
    else:
        m  = st.session_state["foam_morph"]
        ft = st.session_state.get("foam_type", "open")
        E_pred     = gibson_ashby_modulus(m["solid_fraction"], E_solid, ft)
        sig_pred   = gibson_ashby_strength(m["solid_fraction"], sig_solid, ft)
        rho_solid  = mp["rho_gcm3"] if material != "Custom" else 1.24
        rho_eff    = rho_solid * m["solid_fraction"]

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("E* (Young's)", f"{E_pred:.1f} MPa")
        mc2.metric("σ* (compressive)", f"{sig_pred:.2f} MPa")
        mc3.metric("ρ* (effective)", f"{rho_eff:.3f} g/cm³")
        mc4.metric("Specific stiffness", f"{E_pred/max(rho_eff,1e-6):.0f} MPa·cm³/g")

        # Ashby map: E* vs density across porosity range
        st.markdown("#### Ashby map — stiffness vs density")
        phi_range = np.linspace(0.40, 0.98, 120)
        sf_range  = 1.0 - phi_range
        E_open   = np.array([gibson_ashby_modulus(sf, E_solid, "open")   for sf in sf_range])
        E_closed = np.array([gibson_ashby_modulus(sf, E_solid, "closed") for sf in sf_range])
        rho_range = rho_solid * sf_range

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(rho_range, E_open,   color=C_BLUE,  lw=2, label="Open-cell")
        ax.semilogy(rho_range, E_closed, color=C_TEAL,  lw=2, label="Closed-cell")
        ax.axvline(rho_eff, color=C_CORAL, lw=1.5, ls="--")
        ax.axhline(E_pred,  color=C_CORAL, lw=1.5, ls="--")
        ax.scatter([rho_eff], [E_pred], color=C_CORAL, s=80, zorder=5,
                   label=f"This foam ({ft}, E*={E_pred:.1f} MPa)")
        ax.set_xlabel("Effective density ρ* (g/cm³)")
        ax.set_ylabel("Young's modulus E* (MPa, log)")
        ax.set_title(f"Gibson-Ashby map — base: {material}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

        st.info(
            f"**Material: {material}** · E_solid = {E_solid:.0f} MPa, "
            f"σ_solid = {sig_solid:.0f} MPa · "
            f"Solid fraction = {m['solid_fraction']:.3f}"
        )


# ──────────────────────────────────────────────────────────────
# TAB 4 — LOAD REAL FOAM µCT
# ──────────────────────────────────────────────────────────────
with tab_load:
    st.markdown(
        "Upload a binarised foam micro-CT volume (solid = 1, pore = 0) "
        "to measure its morphometrics and overlay them on the Ashby map."
    )
    st.markdown(
        "**Supported formats:** NumPy `.npy`, TIFF stack as `.zip`, single TIFF slice."
    )

    col_fmt, col_vox = st.columns([2, 1])
    real_fmt  = col_fmt.selectbox("Format", [".npy binary mask", "TIFF stack (ZIP)", "Single TIFF"], key="foam_real_fmt")
    real_vox  = col_vox.number_input("Voxel size (µm)", value=39.0, step=1.0, key="foam_real_vox")
    thresh_pct = st.slider("Threshold percentile (for grayscale TIFFs)", 0, 100, 50, 1, key="foam_thresh")

    real_solid = None

    if real_fmt == ".npy binary mask":
        up = st.file_uploader("Upload .npy mask", type=["npy"], key="foam_npy")
        if up:
            arr = np.load(up)
            real_solid = (arr > 0).astype(np.uint8)

    elif real_fmt == "TIFF stack (ZIP)":
        import io, zipfile
        from PIL import Image
        up = st.file_uploader("Upload ZIP of TIFF slices", type=["zip"], key="foam_zip")
        if up:
            slices = []
            with zipfile.ZipFile(io.BytesIO(up.read())) as zf:
                names = sorted(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")))
                for n in names:
                    with zf.open(n) as f:
                        slices.append(np.array(Image.open(f)))
            raw = np.stack(slices, axis=0)
            t   = np.percentile(raw, thresh_pct)
            real_solid = (raw >= t).astype(np.uint8)

    elif real_fmt == "Single TIFF":
        from PIL import Image
        import io
        up = st.file_uploader("Upload TIFF", type=["tif", "tiff"], key="foam_tiff")
        if up:
            raw = np.array(Image.open(up))
            if raw.ndim == 2:
                raw = raw[np.newaxis, ...]
            t = np.percentile(raw, thresh_pct)
            real_solid = (raw >= t).astype(np.uint8)

    if real_solid is not None:
        with st.spinner("Measuring morphometrics…"):
            rm = measure_foam_morphometrics(real_solid, real_vox)
        st.success(f"Measured — {real_solid.shape[2]}×{real_solid.shape[1]}×{real_solid.shape[0]} voxels")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Porosity",    f"{rm['porosity']:.1%}")
        r2.metric("Strut dia",   f"{rm['strut_dia_um']:.0f} µm")
        r3.metric("Cell size",   f"{rm['cell_size_um']:.0f} µm")
        r4.metric("SA density",  f"{rm['sa_density_mm2mm3']:.2f} mm²/mm³")

        E_real = gibson_ashby_modulus(rm["solid_fraction"], E_solid, "open")
        st.metric("Predicted E* (Gibson-Ashby)", f"{E_real:.1f} MPa")

        if st.button("📥 Store as active foam", key="foam_store_real"):
            st.session_state["foam_solid"]    = real_solid
            st.session_state["foam_morph"]    = rm
            st.session_state["foam_type"]     = "open"
            st.session_state["foam_voxel_um"] = real_vox
            st.session_state["foam_E_MPa"]    = E_real
            st.session_state["foam_material"] = material
            st.success("Stored — go to **Connect to pipeline** tab.")
    else:
        st.info("Upload a foam volume above to measure it.")

    st.markdown("---")
    st.markdown(
        "**Tips for foam micro-CT:**\n\n"
        "- Scan at the highest achievable resolution relative to the strut diameter. "
        "Aim for ≥ 3 voxels across the thinnest strut so the distance-transform "
        "thickness measurement is reliable.\n"
        "- Binarise with Otsu's method (or adjust the percentile threshold above) "
        "before uploading; partial-volume artefacts at strut surfaces inflate the "
        "measured SA density.\n"
        "- Keep the ROI cubic if possible — morphometric estimators assume isotropy. "
        "You can check anisotropy using the MIL (mean intercept length) tool in "
        "the ROI Detection page after pushing the mask to session."
    )


# ──────────────────────────────────────────────────────────────
# TAB 5 — CONNECT TO PIPELINE
# ──────────────────────────────────────────────────────────────
with tab_connect:
    st.markdown("### Push foam data to the rest of the tool")

    if "foam_morph" not in st.session_state:
        st.info("Generate or load a foam volume first (tabs 1 or 4).")
    else:
        m  = st.session_state["foam_morph"]
        ft = st.session_state.get("foam_type", "open")
        Ef = st.session_state.get("foam_E_MPa", 0.0)

        st.markdown(
            f"**Active foam:** {ft}-cell · porosity {m['porosity']:.1%} · "
            f"E* ≈ {Ef:.1f} MPa · {st.session_state.get('foam_material','?')}"
        )

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("#### → Generator targets")
            st.markdown(
                "Map foam morphometrics to the bone generator's calibration loop — "
                "useful for generating synthetic porous scaffolds with foam-like topology."
            )
            if st.button("📤 Push to Generator", key="foam_to_gen"):
                # Map foam params to generator session keys (morphometric targets)
                sf = m["solid_fraction"]
                targets = {
                    "target_bvtv":    round(sf, 3),
                    "target_tbth_um": round(m["strut_dia_um"], 1),
                    "target_tbsp_um": round(m["cell_size_um"], 1),
                    "target_tbn_mm":  round(1000.0 / max(m["cell_size_um"], 1), 2),
                    "target_conn_d":  round(m.get("largest_cc_fraction", 0), 3) * 10,
                }
                st.session_state.update(targets)
                st.session_state["foam_morphometrics"] = m
                st.session_state["gen_bvtv"] = targets["target_bvtv"]
                st.session_state["gen_tbth"] = int(targets["target_tbth_um"])
                st.success("Targets pushed — open **Generator** and they will be pre-set.")

        with col_p2:
            st.markdown("#### → FE Solver material")
            st.markdown(
                "Override the FE solver's tissue Young's modulus with the "
                "Gibson-Ashby E* for this foam."
            )
            if st.button("📤 Push E* to FE Solver", key="foam_to_fe"):
                st.session_state["fe_E_tissue_MPa"] = Ef
                st.session_state["fe_material_label"] = (
                    f"{st.session_state.get('foam_material','foam')} foam "
                    f"(E*={Ef:.0f} MPa, ρ*={m['solid_fraction']:.3f})"
                )
                st.success(
                    f"FE solver will use E* = {Ef:.1f} MPa. "
                    "Open **FE Solver** to run the analysis."
                )

        st.markdown("---")
        st.markdown("#### → 3D Viewer")
        if "foam_solid" in st.session_state and st.button("📤 Push to 3D Viewer", key="foam_to_3d"):
            solid = st.session_state["foam_solid"]
            st.session_state["bone_volume"] = {
                "bone_mask": solid,
                "voxel_um":  st.session_state.get("foam_voxel_um", 39.0),
                "bvtv":      float(solid.mean()),
            }
            st.success("Pushed — open **3D Viewer** to inspect the foam mesh.")

        st.markdown("---")
        st.markdown("#### → QKSVM classifier")
        st.markdown(
            "The QKSVM page can classify foam specimens by density class. "
            "Push your foam morphometrics as a pipeline sample:"
        )
        if st.button("📤 Push to QKSVM", key="foam_to_qksvm"):
            st.session_state["morphometrics"] = m
            st.session_state["last_morphometrics"] = m
            st.success("Morphometrics pushed — open **QKSVM Results** to see classification.")

        # Show current session state summary
        with st.expander("Session state summary (foam keys)"):
            foam_keys = {k: v for k, v in st.session_state.items() if "foam" in k.lower()}
            st.json({k: str(v)[:80] for k, v in foam_keys.items()})



# ══════════════════════════════════════════════════════════════
# BATCH PARAMETER SWEEP (appended tab)
# ══════════════════════════════════════════════════════════════
# Note: tabs are already closed above; this adds a new section
# after the tab block as a collapsible expander at page bottom.

st.divider()
st.markdown("## 🔬 Batch parameter sweep")
st.markdown(
    "Sweep porosity × strut scale to map the morphometric and mechanical "
    "landscape. Useful for designing foam specimens before scanning."
)

with st.expander("Run sweep", expanded=False):
    sc1, sc2 = st.columns(2)
    por_min = sc1.slider("Porosity min", 0.50, 0.85, 0.65, 0.05, key="sw_por_min")
    por_max = sc2.slider("Porosity max", 0.60, 0.97, 0.90, 0.05, key="sw_por_max")
    sig_vals_raw = st.multiselect(
        "Strut scales (σ voxels)",
        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        default=[2.0, 3.0, 4.0],
        key="sw_sigs",
    )
    sw_size = st.select_slider("Volume size (vox³)", [24, 32, 40, 48], value=32, key="sw_sz")
    sw_n    = st.slider("Porosity steps", 3, 8, 4, 1, key="sw_n")
    sw_E    = st.number_input("E_solid (MPa)", value=float(E_solid), step=100.0, key="sw_E")

    if st.button("▶ Run sweep", type="primary", key="btn_sweep"):
        import itertools
        por_vals = np.linspace(por_min, por_max, sw_n)
        sig_vals = sig_vals_raw or [3.0]
        rows = []
        prog = st.progress(0.0)
        total = len(por_vals) * len(sig_vals)
        done  = 0
        for por, sig in itertools.product(por_vals, sig_vals):
            solid = generate_open_cell_foam(sw_size, por, sig, seed=42)
            m_s   = measure_foam_morphometrics(solid, voxel_um=39.0)
            E_ga  = gibson_ashby_modulus(m_s["solid_fraction"], sw_E, "open")
            rows.append({
                "Porosity":      round(por, 3),
                "σ (vox)":       sig,
                "Actual por":    round(m_s["porosity"], 3),
                "Strut dia (µm)": round(m_s["strut_dia_um"], 1),
                "Cell size (µm)": round(m_s["cell_size_um"], 1),
                "E* (MPa)":      round(E_ga, 2),
            })
            done += 1
            prog.progress(done / total)
        st.session_state["sweep_rows"] = rows
        prog.empty()

    if "sweep_rows" in st.session_state:
        import pandas as pd
        df = pd.DataFrame(st.session_state["sweep_rows"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Heatmap: E* over porosity × sigma
        sig_vals_used = sorted(df["σ (vox)"].unique())
        por_vals_used = sorted(df["Porosity"].unique())
        if len(sig_vals_used) > 1 and len(por_vals_used) > 1:
            import numpy as _np
            Z = _np.zeros((len(sig_vals_used), len(por_vals_used)))
            for i, sv in enumerate(sig_vals_used):
                for j, pv in enumerate(por_vals_used):
                    sub = df[(df["σ (vox)"] == sv) & (df["Porosity"] == pv)]
                    Z[i, j] = sub["E* (MPa)"].values[0] if len(sub) else 0
            fig, ax = plt.subplots(figsize=(7, 3.5))
            im = ax.imshow(Z, cmap="plasma", aspect="auto",
                           extent=[min(por_vals_used), max(por_vals_used),
                                   min(sig_vals_used)-0.25, max(sig_vals_used)+0.25],
                           origin="lower")
            ax.set_xlabel("Target porosity")
            ax.set_ylabel("Strut scale σ (vox)")
            ax.set_title("E* heatmap (MPa) — open-cell Gibson-Ashby")
            plt.colorbar(im, ax=ax, label="E* (MPa)")
            st.pyplot(fig); plt.close()
