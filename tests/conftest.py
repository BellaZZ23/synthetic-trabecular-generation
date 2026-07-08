"""
tests/conftest.py
=================
Shared fixtures and configuration for the trabecular test suite.

Usage
-----
    pytest                  # terminal output only (fast)
    pytest --plots          # terminal output + saves figures to output/test_plots/
    pytest -v --plots       # verbose terminal + figures

The --plots flag is designed for supervisor demos: run once to generate
a complete set of figures showing what every module does.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Make the repo root importable without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fe_coupling"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── CLI option ────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--plots", action="store_true", default=False,
        help="Save demonstration figures to output/test_plots/",
    )


# ── Plot helper ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def plot_dir(request):
    """Returns Path to output directory if --plots was passed, else None."""
    if request.config.getoption("--plots"):
        d = REPO_ROOT / "output" / "test_plots"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n  [plots → {d}]")
        return d
    return None


@pytest.fixture
def save_plot(plot_dir):
    """
    Fixture that returns a function: save_plot(fig, name).
    Saves figure to output/test_plots/<name>.png when --plots is active,
    then closes it. No-ops silently when --plots is not passed.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for CI
    import matplotlib.pyplot as plt

    def _save(fig, name: str):
        if plot_dir is not None:
            p = plot_dir / f"{name}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"  [saved → {p.name}]")
        plt.close(fig)

    return _save


# ── Shared bone-volume fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_mask(rng):
    """
    Small (16, 32, 32) binary bone mask generated from a GRF zero-crossing.
    Fast enough for DVC and morphometric tests (~0.5 s).
    """
    from scipy.ndimage import gaussian_filter
    shape = (16, 32, 32)
    field = gaussian_filter(rng.standard_normal(shape), sigma=2.5)
    tau   = np.percentile(np.abs(field), 70)   # ~30% BV/TV
    return (np.abs(field) < tau).astype(np.uint8)


@pytest.fixture(scope="session")
def medium_mask(rng):
    """
    Medium (24, 48, 48) mask. Used for measurements and validation tests.
    """
    from scipy.ndimage import gaussian_filter
    shape = (24, 48, 48)
    field = gaussian_filter(rng.standard_normal(shape), sigma=2.8)
    tau   = np.percentile(np.abs(field), 67)
    return (np.abs(field) < tau).astype(np.uint8)


@pytest.fixture(scope="session")
def grayscale_volume(small_mask, rng):
    """Synthetic grayscale µCT volume matching small_mask."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    edt  = distance_transform_edt(small_mask)
    gray = np.where(small_mask, 70 + edt * 3.0, 10.0)
    gray = np.clip(gray + rng.normal(0, 2.0, gray.shape), 0, 255)
    return gray.astype(np.float32)
