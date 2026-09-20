# QCCG × RFQG

**A falsifiable quantum-gravity research program connecting a finite microscopic causal model to a Lorentzian continuum effective theory.**

> Status: active research. This repository does **not** claim a completed theory of quantum gravity, a proven Reuter fixed point, or singularity resolution.

## In one sentence

We are testing whether a finite-dimensional causal quantum system (**QCCG**) can coarse-grain into a Lorentzian gravitational continuum described by the **RFQG-5** / asymptotic-safety program:

```text
QCCG microscopic dynamics
    -> causal / CDT-like coarse geometry
    -> Lorentzian FRG / RFQG effective action
    -> Reuter universality test
    -> real-time strong gravity
```

Every step is treated as a separate obligation. Failed calculations are kept as evidence; they are not tuned away.

## What QCCG is

QCCG starts from finite local quantum registers with a Weyl algebra rather than continuum canonical commutators. Geometry is encoded through causal/cellulation data, local moves, and finite-dimensional operators. The goal is to obtain the gravitational continuum as an emergent critical phase rather than assuming a background metric microscopically.

## What RFQG is

RFQG-5 is the continuum target and audit program. It works with Lorentzian effective dynamics, momentum-dependent vertices/form factors, gauge/background-independence tests, unitarity, UV fixed-point data, and eventually real-time strong-gravity evolution.

## Current status

| Sector | Current result |
| --- | --- |
| Finite microscopic kinematics | Finite Weyl/qudit construction and explicit parent Hamiltonian audits pass at their stated scope. |
| Linearized gravity target | Rank-two TT sector, linearized constraints, and massless tensor scaling pass as continuum-target checks. |
| Causal geometry | Finite 4D foliated simplicial architectures, reversible Pachner moves, slab constructions, and finite connectivity tests are implemented. |
| Formal verification | Selected slab-gluing statements are proved in Lean without imported axioms. |
| Unitarity | Finite microscopic evolution and free/band-limited continuum scaling tests pass; the full interacting graph-changing physical scaling limit remains open. |
| QCCG -> CDT kinetic bridge | The curvature-equilibrated volume generator matches the finite-volume CDT affine kinetic form `D*(N3-n0)` with `D ~ 6.616`, `n0 ~ -0.878`, `R^2 ~ 0.99895` in the audited finite ensemble. |
| CDT potential sector | Regge-curvature weighting improves the CDT-like potential diagnostic, but stable continuum `mu/lambda/omega` extraction is still under active study. |
| QCCG -> Reuter fixed point | **Not established.** CDT/FRG/Reuter results are used only as external anchors until QCCG supplies its own critical trajectory and essential universality data. |

## What is still missing

The main unresolved physics is:

- a controlled large-volume **3+1 Lorentzian continuum phase**;
- interacting stability of a common matter/gravity light cone;
- full nonlinear **BRST / diffeomorphism / hypersurface-deformation closure**;
- an interacting graph-changing physical-Hilbert-space unitarity limit;
- stable CDT-scale effective coefficients and a justified scale map to FRG;
- QCCG essential critical exponents matching the Reuter universality target;
- a real-time CTP/in-in strong-gravity solution for collapse or cosmology.

These are theory-existence / universality tests, not bookkeeping tasks.

## Repository layout

```text
qccg/
  run_*.py              numerical / symbolic / negative-control audits
  theorygate.yaml       claims, obligations, blockers, promotion rules
  SOURCES.md            literature anchors and source policy
  lean/                 Lean proofs
.github/workflows/      research CI and focused probes
MIGRATION.md            provenance of the split from TheoryGate
requirements.txt        pinned audit dependencies
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt

# A small microscopic / quadratic audit
python qccg/run_quadratic_audit.py --out artifacts/quadratic.json

# Validate the full research ledger
theorygate validate qccg/theorygate.yaml

# Lean proofs
cd qccg/lean && lake build
```

Many heavier scripts are Monte Carlo or finite-size scans. GitHub Actions runs them as focused probes so one failure does not hide the rest.

## Audit policy

This project uses [TheoryGate](https://github.com/HeliCorgi/theorygate) as a claim-audit layer.

A `PASS` means only that the explicitly encoded test passed at its stated scope. It does **not** automatically promote a stronger physical claim. Examples:

- a finite causal move set is not a proof of continuum manifold emergence;
- a two-point match is not a proof of nonlinear GR;
- a CDT-like effective action is not a proof of asymptotic safety;
- an external FRG fixed point is a comparison target, not QCCG evidence.

Negative controls and failed approaches are intentionally preserved in the ledger.

## Key references

- T. Regge, *General relativity without coordinates*, Nuovo Cimento 19 (1961), DOI: https://doi.org/10.1007/BF02733251
- J. Ambjørn et al., *The transfer matrix in four-dimensional CDT*, JHEP 09 (2012) 017, arXiv: https://arxiv.org/abs/1205.3791
- J. Ambjørn et al., *The effective action in 4-dim CDT. The transfer matrix approach*, JHEP 06 (2014) 034, arXiv: https://arxiv.org/abs/1403.5940
- J. Ambjørn et al., *Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group*, Phys. Rev. D 110, 126006 (2024), DOI: https://doi.org/10.1103/PhysRevD.110.126006
- A. Baldazzi et al., *Robustness of the derivative expansion in asymptotic safety*, Phys. Rev. D 113, 026005 (2026), arXiv: https://arxiv.org/abs/2312.03831

See [`qccg/SOURCES.md`](qccg/SOURCES.md) for the literature-anchor policy and additional references.

## Provenance

This repository was split from the QCCG research branch of [`HeliCorgi/theorygate`](https://github.com/HeliCorgi/theorygate). See [`MIGRATION.md`](MIGRATION.md) for the exact source commit.
