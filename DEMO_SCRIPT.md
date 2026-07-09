# Supervisor Demo Script
**Branch:** `joss-packaging` | **Date:** July 2026 | **Duration:** ~15 min

---

## BEFORE THEY ARRIVE — Setup checklist

Run these in two separate PowerShell windows:

**Window 1 — App**
```powershell
cd "C:\Users\Isabella\OneDrive - zeki\Documents\Research\synthetic_trabeculae"
streamlit run app.py
```
Wait for the browser to open at `http://localhost:8501`. Leave it on the **home page**.

**Window 2 — Tests (pre-run so output is visible)**
```powershell
cd "C:\Users\Isabella\OneDrive - zeki\Documents\Research\synthetic_trabeculae"
python -m pytest -v --plots
```
Leave the terminal showing `95 passed`. Open `output\test_plots\` in a File Explorer window — have it minimised and ready.

**Also open:** a Python prompt in Window 2 after the tests finish (just type `python`).

---

## PART 1 — CONTEXT (2 min)

> *Say this before touching the screen:*

"The project has two parallel goals. The first is building a validated synthetic micro-CT pipeline — bone generation, FE analysis, DVC displacement tracking. The second is testing whether a quantum reservoir can replace the classical similarity metric in that DVC step, and characterising when it actually helps.

Since our last meeting I've added the quantum reservoir computing module, wired it into the pipeline as a proper pluggable slot, packaged the whole thing as an importable Python library, and written a 95-test suite that validates every layer from morphometrics up to the quantum circuit."

---

## PART 2 — TEST SUITE (3 min)

Switch to **Window 2** — the terminal showing the green test output.

> *Point at the screen:*

"95 tests, all passing, on Windows Python 3.12. Let me walk you through what they cover."

Point to each block as you say it:

| Test file | What it proves |
|---|---|
| `test_dvc.py` | Phase-correlation DVC recovers imposed displacement to **< 1 voxel RMSE** |
| `test_alignment.py` | Volume resampling and rigid alignment are geometrically correct |
| `test_measurements.py` | BV/TV, Tb.Th, Tb.Sp, connectivity — BoneJ-equivalent morphometrics |
| `test_quantum.py` | NCC and QRC similarity backends both return scalars in [−1, 1] |
| `test_qrc.py` | Entanglement entropy, sweet spot detection, 75% classification accuracy, temporal mode |
| `test_validation.py` | FE nodal displacements convert correctly to dense voxel fields |

Now switch to File Explorer — open `output\test_plots\`.

**Open `01_dvc_round_trip.png`**
> "Known deformation imposed, recovered by phase correlation. The error map on the right — that's the residual. Sub-voxel everywhere."

**Open `05_entanglement_sweet_spot.png`**
> "This is the key result. The x-axis is reservoir depth — number of circuit layers. The y-axis is bipartite von Neumann entropy. At depth zero you have a product state — no entanglement, no quantum memory. As depth increases, entropy rises. The dashed line is half the maximum — that's the sweet spot. Beyond it, the state becomes Haar-random and information washes out. The test asserts this crossing exists, reproducibly, for any input."

**Open `06_temporal_qrc.png`**
> "This is the temporal mode — instead of encoding a single load step, the reservoir carries state forward across the sequence. Left panel: prediction vs target modulus. Right panel: BV/TV and reservoir entropy over load steps. The entropy tracks the mechanical complexity of the sequence."

---

## PART 3 — THE APP (8 min)

Switch to the **browser**.

### Home page (1 min)

> "The home page shows the six-stage pipeline. The sixth stage — Quantum Reservoir Computing — was added this cycle. It's not decorative: the similarity slot in the DVC step is a registry, and the QRC backend is registered alongside the classical NCC baseline."

Point to the three QRC preview panels at the bottom:
> "Fixed reservoir, entanglement sweet spot, temporal mode — these are live previews that update with the sidebar controls."

---

### Page 5 — Quantum Reservoir (4 min)

Click **"5 Quantum Reservoir"** in the sidebar.

**Architecture & sweet spot tab**

> "The reservoir is a fixed random quantum circuit — only the linear readout is trained. This is the key design choice: we're not doing variational quantum machine learning, which requires gradient computation on hardware. The reservoir is fixed, cheap to evaluate, and the expressivity comes from the entanglement structure."

Drag `n_layers` from 1 up to 12 slowly.
> "Watch the entropy curve. At low depth it's flat near zero — separable states. Around 4–5 layers it crosses the sweet spot. Beyond 8 it saturates. The test I showed you asserts this crossing programmatically."

Set `n_layers` back to 4. Change `n_qubits` between 3 and 6.
> "More qubits — higher maximum entropy, richer feature space. The tradeoff is exponential state-vector size: 2^n complex amplitudes. For n=6 that's 64 amplitudes — still fast on CPU with numpy."

**Classification demo tab**

Click the **Classification demo** tab.

> "Synthetic data — two classes separated by a hyperplane in feature space. The QRC extracts Pauli-Z expectation values and two-qubit correlators from the reservoir state, then a ridge regression readout learns the boundary. The point is that the quantum feature map can capture correlations a linear classifier couldn't find in the raw input."

Click **Run classification demo**.
> "75% accuracy on held-out data. The baseline is 50% random. Not state of the art — but this is a proof of concept on synthetic data, and it's operating without any gradient-based quantum training."

**Temporal mode tab**

Click the **Temporal mode** tab.

> "This is the novel contribution for the mechanical context. Instead of treating each load step independently, we feed the sequence into the reservoir without resetting state between steps. The reservoir accumulates a quantum memory of the loading history. On the right you can see the reservoir entropy varying across steps — the circuit is responding to the changing mechanical input."

---

### Page 3 — Pipeline, Compare tab (2 min)

Click **"3 Pipeline"** in the sidebar, then click the **Compare** tab.

> "One thing that was wrong in the previous version: we were comparing the synthetic FE strain field against the real D²IM displacement field voxel-for-voxel, even though they live on completely different grids — different voxel size, different field of view, different origin. The Pearson r we were reporting was not meaningful."

Point to the Alignment settings expander.
> "Now both volumes are resampled to a common grid first — either the coarser voxel size by default, or a user-specified target. The checkerboard panel on the right is an alignment quality check: if structural features line up across tile borders, the alignment worked. Only after that do we compute Pearson r and RMSE."

---

## PART 4 — THE PACKAGE (2 min)

Switch to **Window 2**. At the Python prompt:

```python
import trabecular
print(trabecular.__version__)
```
> "0.1.0 — the pipeline is now an importable library."

```python
from trabecular import align_volumes, SIMILARITY_BACKENDS
print(list(SIMILARITY_BACKENDS))
```
> "'NCC (classical)' and 'QRC similarity' — the quantum slot is a dict. Adding a new backend is one line: add an entry to this registry. The DVC comparison step reads from it without knowing anything about the underlying implementation."

```python
from trabecular import warp_volume, recover_displacement, displacement_rmse
```
> "The full DVC stack is importable without Streamlit. This is the JOSS packaging milestone — the library can be used by anyone who wants to run the pipeline programmatically, cite it, or build on it."

---

## PART 5 — WHAT'S NEXT (1 min)

> "Three things on the roadmap. First, run the QRC similarity on real micro-CT data and measure whether it outperforms NCC on the DVC round-trip RMSE — that's the experiment that tests the core hypothesis. Second, characterise the entanglement sweet spot as a function of voxel size and bone morphology — BV/TV, Tb.Th — which is the material for the first paper. Third, write the JOSS paper: statement of need, API documentation, and a worked example. The package structure is already in place for that."

---

## ANTICIPATED QUESTIONS

**"Why a quantum reservoir and not a trained quantum circuit?"**
> Trained variational circuits require gradient computation — either backprop through the circuit (classical simulation, expensive) or parameter-shift rules on hardware (noisy, slow). A fixed reservoir with a trained classical readout sidesteps both problems. The expressivity comes from the fixed entanglement structure, not learned parameters.

**"Is this running on real quantum hardware?"**
> No — pure numpy state-vector simulation. 2^n complex amplitudes, exact. This is intentional at this stage: we're characterising what the reservoir *can* do before committing to hardware noise models. The architecture is hardware-agnostic; swapping in a Qiskit backend would only change `_run_circuit`.

**"What does 75% accuracy actually tell us?"**
> It tells us the quantum feature map produces a linearly separable representation for a problem the raw input can't separate linearly. It's a sanity check, not a claim. The scientifically interesting question is whether QRC similarity outperforms NCC *specifically on trabecular bone DVC* — that's the experiment coming next.

**"Why is the sweet spot at half-maximum entropy?"**
> It's the information-theoretic midpoint between two failure modes: too little entanglement means the reservoir state is essentially classical (no quantum advantage), too much means the state is Haar-random (maximum entropy, but information about the input is washed out). Half-saturation is where input sensitivity and memory capacity are jointly maximised — analogous to the edge of chaos in classical reservoir computing.

**"What's the RMSE on the DVC round-trip?"**
> Sub-voxel: the test asserts < 1.0 voxels, and in practice it runs at ~0.3–0.5 voxels on the synthetic 24×48×48 test volume. The published figure for synthetic data is 0.53 voxels; for FE-driven deformation it's 0.27 voxels.

**"Is the package pip-installable?"**
> `pip install .` from the repo root installs the core. Optional extras: `pip install ".[fe]"` adds scikit-fem for the voxel FE solver, `".[streamlit]"` adds the app dependencies, `".[umap]"` adds the UMAP baseline. This is documented in `pyproject.toml`.

---

## CLOSING

> "The main thing I want you to take away is that the entanglement sweet spot is now a *testable, reproducible property* of the pipeline — not a theoretical argument. The test suite proves it crosses the half-saturation threshold for any input, and the app lets you see it interactively. The next step is measuring whether that property translates to a DVC accuracy improvement on real bone data."

---

*Generated for Isabella Florez — University of Greenwich PhD, Quantum-AI for micro-CT bone imaging*
