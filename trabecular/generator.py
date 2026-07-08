"""
trabecular.generator
====================
Synthetic trabecular bone generator — public API wrapper.

The canonical implementation lives in ``fe_coupling/step3_generator_fe_coupling.py``
(which itself calls ``scripts/synthetic_trabecular_v15_morphometric_control.py``).
This module re-exports the three generator functions under the package namespace
so users only ever need::

    from trabecular.generator import generate, generate_calibrated, make_grayscale

Full refactor of the generator into this module is tracked as a separate task.

Functions
---------
generate(...)              Generate a bone volume (BV/TV target)
generate_calibrated(...)   Generate with iterative Tb.Th calibration
make_grayscale(...)        Synthesise a µCT grayscale from a binary mask
"""
from __future__ import annotations

from trabecular.fe.coupling import (
    generate_bone_volume      as generate,           # noqa: F401
    generate_bone_volume_calibrated as generate_calibrated,  # noqa: F401
    generate_grayscale        as make_grayscale,     # noqa: F401
)

__all__ = ["generate", "generate_calibrated", "make_grayscale"]
