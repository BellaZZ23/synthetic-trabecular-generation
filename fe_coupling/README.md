# FE Coupling Module

Mechanically-aware generator coupling with micro-FE solver.

## Scripts
- step1_patch_test.py   -- Single-element patch test validation
- step2_voxel_fe.py     -- Voxel-to-hex mesh conversion + uniaxial compression
- step3_generator_coupling.py -- Couple v15 generator with FE solver (TODO)

## Setup
python -m venv femenv
femenv\Scripts\activate
pip install scikit-fem numpy scipy matplotlib
