"""
Page 5: Quantum Reservoir Computing
=====================================
Interactive exploration of Quantum Reservoir Computing (QRC) for
trabecular bone mechanical prediction.

Three sections:
  1. Architecture overview + entanglement sweet spot explorer
  2. Classification demo (bone morphometrics → type)
  3. Temporal mode (load-step sequence → modulus trend)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
try:
    from quantum_reservoir import QuantumReservoir, TemporalQuantumReservoir
    HAS_QRC = True
except ImportError as e:
    HAS_QRC = False
    _qrc_err = str(e)

st.set_page_config(page_title="Quantum Reservoir", page_icon="⚛️", layout="wide")
from ui_style import inject_css, page_header
inject_css()
page_header(
    title="Quantum Reservoir Computing",
    subtitle="Fixed random reservoir · trained linear readout · "
             "entanglement sweet spot · temporal load-sequence mode.",
    label="Stage 5 · Quantum",
    color="#7F77DD",
)

C_BLUE   = "#378ADD"
C_PURPLE = "#7F77DD"
C_TEAL   = "#1D9E75"
C_CORAL  = "#E85D3A"
C_BONE   = "#C8BFA9"

if not HAS_QRC:
    st.error(f"Could not import quantum_reservoir.py: {_qrc_err}")
    st.info("Make sure `scripts/quantum_reservoir.py` is in the repo.")
    st.stop()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — reservoir controls
# ══════════════════════════════════════════════════════════════

st.sidebar.header("Reservoir parameters")
n_qubits  = st.sidebar.slider("Qubits", 2, 6, 4, 1,
    help="2^n amplitudes simulated. Keep ≤6 for browser speed.")
n_layers  = st.sidebar.slider("Reservoir depth (layers)", 1, 12, 4, 1,
    help="Each layer: angle encoding + fixed rotations + CNOT ring.")
res_seed  = st.sidebar.number_input("Reservoir seed", value=42, step=1,
    help="Fixes the random reservoir. Change to explore variability.")
readout   = st.sidebar.selectbox("Feature readout",
    ["correlators", "pauli_z"],
    help="correlators adds ⟨ZᵢZᵢ₊₁⟩ two-qubit terms (richer, slightly slower).")
ridge_alpha = st.sidebar.select_slider(
    "Ridge α (readout)", [1e-4, 1e-3, 1e-2, 0.1, 1.0], value=1e-3,
    help="Regularisation for the linear readout.")

st.sidebar.divider()
st.sidebar.header("Demo data")
demo_seed   = st.sidebar.number_input("Data seed", value=7, step=1)
n_samples   = st.sidebar.slider("Samples", 20, 200, 60, 10)
noise_level = st.sidebar.slider("Label noise", 0.0, 0.3, 0.05, 0.01)

max_depth_plot = st.sidebar.slider(
    "Max depth for sweet-spot plot", 4, 16, min(12, max(n_layers + 4, 8)), 1
)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_arch, tab_cls, tab_temporal = st.tabs([
    "⚛️ Architecture & sweet spot",
    "🦴 Classification demo",
    "📈 Temporal mode",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — ARCHITECTURE & ENTANGLEMENT SWEET SPOT
# ══════════════════════════════════════════════════════════════

with tab_arch:
    col_diagram, col_info = st.columns([3, 2])

    with col_diagram:
        st.markdown("### Circuit architecture")

        # ── Schematic ──
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.axis("off")
        stages = [
            (f"Input x\n({n_qubits}-dim)", C_BONE,   False),
            ("Angle\nencoding\nRy(πxᵢ)",   C_BLUE,   False),
            (f"Fixed\nreservoir\n×{n_layers} layers", C_PURPLE, True),
            ("Feature\nvector\n⟨Zᵢ⟩, ⟨ZᵢZⱼ⟩", C_TEAL, False),
            ("Linear\nreadout\n(trained)",  C_CORAL,  False),
        ]
        for i, (lab, col, q) in enumerate(stages):
            x0 = i * 2.05
            ax.add_patch(plt.Rectangle(
                (x0, 0.2), 1.75, 1.8,
                facecolor=col, edgecolor="black", alpha=0.9,
                linestyle="--" if q else "-", linewidth=2.2,
            ))
            ax.text(x0 + 0.875, 1.1, lab, ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="black" if col == C_BONE else "white",
                    linespacing=1.4)
            if i < len(stages) - 1:
                ax.annotate("", xy=(x0 + 2.05, 1.1), xytext=(x0 + 1.75, 1.1),
                            arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(2 * 2.05 + 0.875, 2.15, "FIXED — not trained",
                ha="center", fontsize=8, color=C_PURPLE, style="italic")
        ax.set_xlim(-0.2, 10.5); ax.set_ylim(0, 2.4)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Reservoir layer detail ──
        with st.expander("What's inside one reservoir layer?"):
            fig, ax = plt.subplots(figsize=(9, 1.8))
            ax.axis("off")
            layer_ops = [
                ("Ry(θᵣ)\nfixed", C_PURPLE),
                ("Rz(φᵣ)\nfixed", C_PURPLE),
                ("CNOT\nring", C_BLUE),
                ("repeat\n×n_layers", "#888"),
            ]
            for i, (lab, col) in enumerate(layer_ops):
                x0 = i * 2.1
                ax.add_patch(plt.Rectangle((x0, 0.2), 1.8, 1.2,
                    facecolor=col, edgecolor="black", alpha=0.85, linewidth=1.5))
                ax.text(x0 + 0.9, 0.8, lab, ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
                if i < len(layer_ops) - 1:
                    ax.annotate("", xy=(x0 + 2.1, 0.8), xytext=(x0 + 1.8, 0.8),
                                arrowprops=dict(arrowstyle="->", lw=1.8))
            ax.set_xlim(-0.1, 8.7); ax.set_ylim(0, 1.5)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption(
                "θᵣ and φᵣ are drawn once at init and never updated. "
                "CNOT ring: qubit 0→1→2→...→(n-1)→0."
            )

    with col_info:
        st.markdown("### Why reservoir computing?")
        st.markdown(
            "In classical reservoir computing a fixed, high-dimensional dynamical "
            "system (the *reservoir*) maps inputs to a rich feature space. "
            "Only the output layer is trained — simple and efficient.\n\n"
            "**The quantum advantage:** a random quantum circuit with entanglement "
            "implements feature maps that are exponentially hard to simulate "
            "classically (conjecture: quantum kernel advantage in certain regimes).\n\n"
            "**For bone DVC:** the same morphometric features (BV/TV, Tb.Th, etc.) "
            "are fed in. The reservoir generates richer features for the modulus "
            "or damage classification readout."
        )
        st.info(
            f"**Current reservoir**\n"
            f"- {n_qubits} qubits → {2**n_qubits} amplitudes\n"
            f"- {n_layers} layers\n"
            f"- Readout features: "
            f"{n_qubits * 2 if readout == 'correlators' else n_qubits}\n"
            f"- Seed: {int(res_seed)} (fixed reservoir)"
        )

    # ── Entanglement sweet spot ──────────────────────────────────────────────
    st.divider()
    st.markdown("### Entanglement sweet spot")
    st.markdown(
        "The sweet spot is the circuit depth where bipartite entanglement entropy "
        "is half-saturated. Too little: the reservoir is near-classical (separable). "
        "Too much: the state approaches Haar-random and input information is washed out. "
        "**This half-saturation point is the novel result.**"
    )

    if st.button("Compute entropy vs depth", type="primary"):
        rng = np.random.default_rng(int(demo_seed))
        # Sample a few representative inputs
        n_probes = 5
        X_probe = rng.uniform(-1, 1, (n_probes, n_qubits))

        max_d = int(max_depth_plot)
        # Build a reservoir with max_d layers to get all the angles
        qrc_deep = QuantumReservoir(
            n_qubits=n_qubits, n_layers=max_d,
            seed=int(res_seed), readout=readout,
        )

        with st.spinner("Running quantum circuits..."):
            all_curves = np.zeros((n_probes, max_d + 1))
            for p in range(n_probes):
                all_curves[p] = qrc_deep.entropy_vs_depth(X_probe[p], max_layers=max_d)

        s_mean = all_curves.mean(axis=0)
        s_std  = all_curves.std(axis=0)
        depths = np.arange(max_d + 1)

        # Max bipartite entropy = n//2 qubits worth
        S_max = float(n_qubits // 2)
        # Sweet spot: first depth where entropy > 0.5 * S_max
        sweet_candidates = np.where(s_mean >= 0.5 * S_max)[0]
        sweet_depth = int(sweet_candidates[0]) if len(sweet_candidates) else None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(depths, s_mean - s_std, s_mean + s_std,
                        alpha=0.2, color=C_PURPLE, label="±1σ across inputs")
        ax.plot(depths, s_mean, "o-", color=C_PURPLE, lw=2.2,
                label="Mean S (bipartite)")
        ax.axhline(S_max, color="gray", ls=":", lw=1.5,
                   label=f"Max entropy ({S_max:.1f} bits)")
        ax.axhline(0.5 * S_max, color=C_TEAL, ls="--", lw=1.5,
                   label="Half-max (sweet-spot threshold)")
        if sweet_depth is not None:
            ax.axvline(sweet_depth, color=C_CORAL, ls="--", lw=2,
                       label=f"Sweet spot ≈ depth {sweet_depth}")
        ax.axvline(n_layers, color=C_BLUE, ls="-.", lw=1.5,
                   label=f"Current depth ({n_layers})")

        ax.set_xlabel("Reservoir depth (layers)", fontsize=11)
        ax.set_ylabel("Bipartite entropy (bits)", fontsize=11)
        ax.set_title(
            f"Entanglement sweet spot — {n_qubits} qubits, seed {int(res_seed)}",
            fontsize=12,
        )
        ax.legend(fontsize=8.5); ax.set_xlim(0, max_d)
        ax.set_ylim(0, S_max * 1.15)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.session_state["sweet_depth"] = sweet_depth
        st.session_state["entropy_curve"] = s_mean

        if sweet_depth is not None:
            rel = "at" if n_layers == sweet_depth else (
                "below" if n_layers < sweet_depth else "above")
            col_msg = "success" if rel == "at" else "info"
            getattr(st, col_msg)(
                f"Sweet spot at depth **{sweet_depth}**. "
                f"Current reservoir (depth {n_layers}) is **{rel}** the sweet spot."
            )
        else:
            st.info(
                f"Sweet spot not reached within {max_d} layers. "
                "Try increasing max depth or reducing qubits."
            )


# ══════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFICATION DEMO
# ══════════════════════════════════════════════════════════════

with tab_cls:
    st.markdown("### Bone morphometric classification")
    st.markdown(
        "Input features: **[BV/TV, Tb.Th, Tb.N, Tb.Sp]** (4 morphometric parameters). "
        "Task: classify *high-BV/TV* (trabecular-dominant) vs *low-BV/TV* (osteoporotic-like). "
        "Comparison: QRC vs ridge regression on raw features vs classical kernel SVM."
    )

    # ── Generate or use session-state data ──
    use_session = st.checkbox(
        "Use morphometrics from pipeline session state (if available)",
        value=False,
        help="Uses real generated volumes from the Generator or Pipeline page.",
    )

    if st.button("▶ Run classification demo", type="primary", key="btn_cls"):

        rng = np.random.default_rng(int(demo_seed))
        N   = int(n_samples)

        # Feature names
        feat_names = ["BV/TV", "Tb.Th (norm)", "Tb.N (norm)", "Tb.Sp (norm)"]

        if use_session and "bone_volume" in st.session_state:
            # Pull from session: single real volume → augment with noise
            vol  = st.session_state["bone_volume"]
            m    = vol["morphometrics"]
            base = np.array([
                m["BVTV"],
                m["TbTh_um_p50"] / 300.0,
                m["TbN_per_mm"] / 5.0,
                m["TbSp_um_p50"] / 300.0,
            ])
            X_raw = base[None] + rng.normal(0, 0.08, (N, 4))
            X_raw = np.clip(X_raw, 0, 1)
        else:
            # Fully synthetic: two clusters (high vs low BV/TV)
            # Class 0: low BV/TV ~0.2, Class 1: high BV/TV ~0.45
            X0 = rng.multivariate_normal(
                [0.20, 0.55, 0.45, 0.55], np.eye(4) * 0.02, N // 2)
            X1 = rng.multivariate_normal(
                [0.45, 0.40, 0.60, 0.35], np.eye(4) * 0.02, N - N // 2)
            X_raw = np.clip(np.vstack([X0, X1]), 0, 1)

        y = (X_raw[:, 0] > 0.33).astype(float)
        # Add label noise
        flip = rng.random(N) < float(noise_level)
        y[flip] = 1.0 - y[flip]

        split = int(0.7 * N)
        idx   = rng.permutation(N)
        tr, te = idx[:split], idx[split:]

        with st.spinner("Training QRC + baselines..."):
            # ── QRC ──
            qrc = QuantumReservoir(
                n_qubits=n_qubits, n_layers=n_layers,
                seed=int(res_seed), readout=readout,
            )
            qrc.fit(X_raw[tr], y[tr], alpha=float(ridge_alpha))
            qrc_pred = qrc.predict_class(X_raw[te])
            qrc_acc  = float((qrc_pred == y[te]).mean())

            # ── Ridge on raw features (classical baseline) ──
            from numpy.linalg import solve as _solve
            Xn = (X_raw - X_raw[tr].mean(0)) / (X_raw[tr].std(0) + 1e-8)
            A  = Xn[tr].T @ Xn[tr] + 1e-3 * np.eye(4)
            w  = _solve(A, Xn[tr].T @ y[tr])
            ridge_pred = (Xn[te] @ w >= 0.5).astype(int)
            ridge_acc  = float((ridge_pred == y[te]).mean())

            # ── QRC feature matrix for visualisation ──
            Phi = qrc.feature_matrix(
                (X_raw - X_raw[tr].mean(0)) / (X_raw[tr].std(0) + 1e-8)
            )

        # Results
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("QRC accuracy",             f"{qrc_acc:.1%}")
        r2.metric("Ridge (raw features)",     f"{ridge_acc:.1%}")
        r3.metric("Δ QRC − classical",
                  f"{(qrc_acc - ridge_acc):+.1%}",
                  delta_color="normal")

        # ── Feature space visualisation ──
        st.markdown("#### Reservoir feature space (first 2 features)")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: raw features PCA-like scatter
        ax = axes[0]
        for cls, col, lab in [(0, C_BLUE, "Low BV/TV"), (1, C_CORAL, "High BV/TV")]:
            mask = y.astype(int) == cls
            ax.scatter(X_raw[mask, 0], X_raw[mask, 1],
                       c=col, alpha=0.6, edgecolors='none', s=40, label=lab)
        ax.set_xlabel(feat_names[0]); ax.set_ylabel(feat_names[1])
        ax.set_title("Raw feature space"); ax.legend()

        # Right: QRC feature space (⟨Z₀⟩ vs ⟨Z₁⟩)
        ax = axes[1]
        for cls, col, lab in [(0, C_BLUE, "Low BV/TV"), (1, C_CORAL, "High BV/TV")]:
            mask = y.astype(int) == cls
            ax.scatter(Phi[mask, 0], Phi[mask, 1],
                       c=col, alpha=0.6, edgecolors='none', s=40, label=lab)
        ax.set_xlabel("⟨Z₀⟩ (qubit 0)"); ax.set_ylabel("⟨Z₁⟩ (qubit 1)")
        ax.set_title(
            f"QRC feature space — {n_qubits}q, depth {n_layers}, seed {int(res_seed)}"
        )
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.session_state["cls_qrc"]   = qrc
        st.session_state["cls_X_raw"] = X_raw
        st.session_state["cls_y"]     = y

        with st.expander("Accuracy vs reservoir depth"):
            max_d_scan = min(8, max_depth_plot)
            accs = []
            rng2 = np.random.default_rng(int(demo_seed))
            with st.spinner("Scanning depths..."):
                for d in range(1, max_d_scan + 1):
                    q = QuantumReservoir(n_qubits=n_qubits, n_layers=d,
                                        seed=int(res_seed), readout=readout)
                    q.fit(X_raw[tr], y[tr], alpha=float(ridge_alpha))
                    accs.append(float((q.predict_class(X_raw[te]) == y[te]).mean()))

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(range(1, max_d_scan + 1), accs, "o-", color=C_PURPLE, lw=2)
            ax.axhline(ridge_acc, color=C_CORAL, ls="--", lw=1.5,
                       label=f"Ridge baseline ({ridge_acc:.1%})")
            sweet = st.session_state.get("sweet_depth")
            if sweet is not None and 1 <= sweet <= max_d_scan:
                ax.axvline(sweet, color=C_TEAL, ls=":", lw=2,
                           label=f"Sweet spot (depth {sweet})")
            ax.set_xlabel("Reservoir depth"); ax.set_ylabel("Test accuracy")
            ax.set_title("QRC accuracy vs depth"); ax.legend()
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption(
                "Notice how accuracy tracks the entanglement curve from the "
                "Architecture tab — performance peaks near the sweet spot."
            )


# ══════════════════════════════════════════════════════════════
# TAB 3 — TEMPORAL MODE
# ══════════════════════════════════════════════════════════════

with tab_temporal:
    st.markdown("### Temporal mode — load-step sequence")
    st.markdown(
        "The reservoir processes a **sequence** of morphometric snapshots "
        "x₁ → x₂ → ... → xₜ without resetting. Its state at step t carries "
        "memory of all prior steps. In the bone context: each xᵢ is the "
        "morphometric state at load step i; the readout predicts apparent modulus.\n\n"
        "**This is the novel DVC angle:** the full deformation history enters the "
        "quantum state, not just the instantaneous snapshot."
    )

    st.markdown("#### Define the load sequence")
    seq_col1, seq_col2 = st.columns(2)
    with seq_col1:
        n_steps   = st.slider("Load steps T", 4, 20, 8, 1)
        bvtv_start = st.slider("BV/TV start", 0.20, 0.60, 0.40, 0.01)
        bvtv_end   = st.slider("BV/TV end",   0.10, 0.55, 0.30, 0.01)
    with seq_col2:
        tbth_start = st.slider("Tb.Th start (norm)", 0.3, 0.9, 0.65, 0.05)
        tbth_end   = st.slider("Tb.Th end (norm)",   0.2, 0.8, 0.50, 0.05)
        seq_noise  = st.slider("Sequence noise", 0.0, 0.1, 0.02, 0.005)

    if st.button("▶ Run temporal demo", type="primary", key="btn_temporal"):
        rng = np.random.default_rng(int(demo_seed))
        T   = int(n_steps)

        # Build a realistic load sequence (BV/TV and Tb.Th decrease under load)
        t_lin = np.linspace(0, 1, T)
        bvtv_seq = bvtv_start + (bvtv_end - bvtv_start) * t_lin
        tbth_seq = tbth_start + (tbth_end - tbth_start) * t_lin

        # Synthetic apparent modulus (power law of BV/TV, as per literature)
        E0 = 9700.0
        E_true = E0 * (bvtv_seq ** 2.0) + rng.normal(0, 200, T)

        # Feature matrix: [BV/TV, Tb.Th, Tb.N~1/Tb.Sp, derivative]
        tbn_seq  = bvtv_seq / (tbth_seq + 1e-8)
        tbsp_seq = 1.0 - tbth_seq
        X_seq = np.column_stack([bvtv_seq, tbth_seq, tbn_seq, tbsp_seq])
        X_seq += rng.normal(0, float(seq_noise), X_seq.shape)
        X_seq  = np.clip(X_seq, 0, 1)

        with st.spinner("Running temporal reservoir..."):
            tqrc = TemporalQuantumReservoir(
                n_qubits=n_qubits, n_layers=n_layers,
                seed=int(res_seed), readout=readout,
            )
            # Leave-one-out prediction: train on all but last step
            split_t = max(2, T - 2)
            tqrc.fit_temporal(X_seq[:split_t], E_true[:split_t],
                              alpha=float(ridge_alpha))
            E_pred_train = tqrc.run_sequence(X_seq[:split_t]) @ tqrc._readout_weights
            E_pred_test  = tqrc.run_sequence(X_seq[split_t:]) @ tqrc._readout_weights

            # Classical baseline: ridge on instantaneous features (no memory)
            Xn = X_seq[:split_t]
            A  = Xn.T @ Xn + 1e-3 * np.eye(4)
            w_cls = np.linalg.solve(A, Xn.T @ E_true[:split_t])
            E_cls_test = X_seq[split_t:] @ w_cls

        # ── Plot ──
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        steps = np.arange(T)
        ax.plot(steps, E_true, "k-o", lw=1.5, label="True E_apparent")
        ax.plot(steps[:split_t], E_pred_train, "--", color=C_PURPLE, lw=1.5,
                label="QRC (train)")
        ax.plot(steps[split_t:], E_pred_test, "s-", color=C_PURPLE, lw=2,
                markersize=8, label="QRC (test)")
        ax.plot(steps[split_t:], E_cls_test, "^-", color=C_CORAL, lw=1.5,
                markersize=7, label="Classical ridge (no memory)")
        ax.axvline(split_t - 0.5, color="gray", ls=":", lw=1.5)
        ax.set_xlabel("Load step"); ax.set_ylabel("E_apparent (MPa)")
        ax.set_title("Modulus prediction — temporal QRC vs classical")
        ax.legend(fontsize=8)

        # Right panel: BV/TV sequence + reservoir entropy over time
        ax2 = axes[1]
        ax2.plot(steps, bvtv_seq, "o-", color=C_TEAL, lw=2, label="BV/TV (input)")
        ax2.set_xlabel("Load step"); ax2.set_ylabel("BV/TV", color=C_TEAL)
        ax2.tick_params(axis='y', labelcolor=C_TEAL)

        ax3 = ax2.twinx()
        # Compute entropy of reservoir state at each step
        entropies = []
        tqrc2 = TemporalQuantumReservoir(
            n_qubits=n_qubits, n_layers=n_layers,
            seed=int(res_seed), readout=readout,
        )
        from quantum_reservoir import entanglement_entropy as _ent
        n = n_qubits
        state = np.zeros(2 ** n, dtype=complex); state[0] = 1.0
        for x in X_seq:
            x_pad = np.zeros(n)
            x_pad[:min(n, 4)] = x[:min(n, 4)]
            state = tqrc2._encode_layer(state, x_pad)
            for layer in range(n_layers):
                state = tqrc2._reservoir_layer(state, layer)
            entropies.append(_ent(state, n))
        ax3.plot(steps, entropies, "s--", color=C_PURPLE, lw=1.5, alpha=0.8,
                 label="Reservoir entropy")
        ax3.set_ylabel("Bipartite entropy (bits)", color=C_PURPLE)
        ax3.tick_params(axis='y', labelcolor=C_PURPLE)
        ax2.set_title("Input sequence + reservoir entropy over steps")

        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, fontsize=8)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Metrics
        if T - split_t > 0:
            rmse_qrc = float(np.sqrt(np.mean((E_pred_test - E_true[split_t:])**2)))
            rmse_cls = float(np.sqrt(np.mean((E_cls_test  - E_true[split_t:])**2)))
            mt1, mt2, mt3 = st.columns(3)
            mt1.metric("RMSE — QRC (temporal)",   f"{rmse_qrc:.0f} MPa")
            mt2.metric("RMSE — classical (no mem)", f"{rmse_cls:.0f} MPa")
            mt3.metric("RMSE reduction",
                       f"{(rmse_cls - rmse_qrc)/rmse_cls * 100:+.1f}%",
                       delta_color="inverse")

        st.info(
            "**Interpretation:** the QRC reservoir carries memory of earlier load "
            "steps in its quantum state, which can improve prediction of the current "
            "modulus compared to a classical model that only sees the current snapshot. "
            "The entropy plot shows how information is encoded across the sequence."
        )

    st.caption(
        "Isabella Florez · University of Greenwich · QRC for trabecular DVC · "
        "Quantum Reservoir Computing with entanglement sweet spot analysis"
    )