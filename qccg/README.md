# Research code

This directory contains the executable QCCG/RFQG research program.

Most `run_*.py` files are deliberately narrow audits: one calculation, one failure condition, one evidence JSON output. Stronger claims live in `theorygate.yaml` and are promoted only when all required obligations pass.

## Suggested reading order

1. `run_quadratic_audit.py` — finite Weyl / quadratic tensor pilot
2. `run_parent_hamiltonian_audit.py` — explicit finite-Weyl parent -> Hessian
3. `run_linearized_constraint_audit.py` — linearized spin-2 target
4. `run_causal_foliation_audit.py` and `run_time_local_slice_transfer_toy.py` — causal geometry
5. `run_qccg_cdt_transfer_match.py` — QCCG -> CDT transfer comparison
6. `run_qccg_cdt_affine_kinetic_audit.py` — finite-volume CDT kinetic denominator
7. `run_cdt_frg_anchor_audit.py` and `run_rg_bridge_audit.py` — CDT/FRG/Reuter bridge
8. `theorygate.yaml` — full obligation / blocker ledger

Lean proofs are under `lean/`; literature anchors are in `SOURCES.md`.

For the project overview, current status, and non-claims, start with the repository root [`README.md`](../README.md).
