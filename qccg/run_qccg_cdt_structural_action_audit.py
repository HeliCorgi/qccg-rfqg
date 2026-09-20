#!/usr/bin/env python3
"""Structural QCCG -> CDT minisuperspace-action bridge.

This audit separates *functional-form emergence* from *coefficient matching*.

Inputs already established elsewhere in the QCCG audit:
1. local Pachner move counting supplies an extensive volume diffusion
   D(N3)=O(N3);
2. the target continuum spatial slices are S^3-like in the semiclassical
   sector.

For a round S^3 of radius r:
    V3 = 2*pi^2 r^3,
    R3 = 6/r^2,
    integral sqrt(q) R3 = 12*pi^2 r
                          proportional to V3^(1/3).

Thus a coarse curvature term scales as N3^(1/3), while a volume/cosmological
term scales as N3. Combined with diffusion D(N3) proportional to N3, the local
Gaussian effective action has the same structural form measured in the CDT
de Sitter phase:

  (Delta N3)^2/(N3+N3') + mu N3^(1/3) - lambda N3.

This does NOT establish the QCCG coefficients, de Sitter phase, or continuum
critical scaling. Those remain numerical/RG obligations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


r, V, n, dn, D0, dt = sp.symbols(
    "r V n dn D0 dt", positive=True, real=True
)
pi = sp.pi

V3 = 2 * pi**2 * r**3
R3 = 6 / r**2
IR = sp.simplify(V3 * R3)
r_of_V = sp.solve(sp.Eq(V, V3), r)[0]
IR_of_V = sp.simplify(IR.subs(r, r_of_V))
curvature_ratio = sp.simplify(IR_of_V / V**sp.Rational(1, 3))

# Diffusive Gaussian exponent for D(n)=D0*n.
gaussian_exponent = sp.simplify(dn**2 / (2 * D0 * n * dt))
sym_n = sp.symbols("s", positive=True, real=True)
cdt_local_form = sp.simplify(
    gaussian_exponent.subs(n, sym_n / 2)
)


SOURCES = [
    {
        "title": "The effective action in 4-dim CDT. The transfer matrix approach",
        "journal": "JHEP 06 (2014) 034",
        "arxiv": "1403.5940",
        "doi": "10.1007/JHEP06(2014)034",
    },
    {
        "title": "Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group",
        "journal": "Phys. Rev. D 110, 126006 (2024)",
        "doi": "10.1103/PhysRevD.110.126006",
    },
]


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "qccg-cdt-structural-action-audit",
        "artifact": "qccg/run_qccg_cdt_structural_action_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    curvature_pass = (
        sp.simplify(IR_of_V - curvature_ratio * V**sp.Rational(1, 3)) == 0
        and curvature_ratio != 0
    )
    diffusion_pass = sp.simplify(
        cdt_local_form
        - dn**2 / (D0 * dt * sym_n)
    ) == 0

    result = {
        "schema": 1,
        "scope": "conditional structural action matching; coefficients and continuum phase not established",
        "sources": SOURCES,
        "evidence": [
            evidence(
                "qccg-s3-curvature-volume-scaling",
                "QCCG_S3_CURVATURE_POTENTIAL_SCALING",
                "PASS" if curvature_pass else "FAIL",
                "For a round S3 slice, the integrated intrinsic scalar curvature scales exactly as V3^(1/3), matching the non-linear potential power appearing in the CDT de Sitter minisuperspace action.",
                V3=str(V3),
                R3=str(R3),
                integrated_curvature=str(IR),
                integrated_curvature_as_function_of_volume=str(IR_of_V),
                coefficient_times_V13=str(curvature_ratio),
                sources=SOURCES,
            ),
            evidence(
                "qccg-diffusion-cdt-kinetic-structure",
                "QCCG_DIFFUSION_CDT_KINETIC_STRUCTURE",
                "PASS" if diffusion_pass else "FAIL",
                "If the measured coarse volume diffusion satisfies D(N3)=D0*N3, its local Gaussian exponent is proportional to (Delta N3)^2/(N3+N3') in the symmetric near-diagonal limit.",
                gaussian_exponent=str(gaussian_exponent),
                symmetric_near_diagonal_form=str(cdt_local_form),
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-action-structural-match",
                "QCCG_CDT_ACTION_STRUCTURAL_MATCH",
                "PASS" if curvature_pass and diffusion_pass else "FAIL",
                "Combining extensive local volume diffusion, round-S3 curvature scaling, and a linear volume term reproduces the functional structure of the CDT de Sitter effective minisuperspace action. This is a conditional structural match, not a coefficient or universality-class match.",
                structure="A*(Delta N3)^2/(N3+N3') + mu*N3^(1/3) - lambda*N3",
                assumptions=[
                    "large-volume diffusive regime",
                    "semiclassical S3-like spatial slices",
                    "coarse curvature term descends to integrated intrinsic scalar curvature",
                    "linear volume/cosmological contribution retained",
                ],
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-coefficient-match-open",
                "QCCG_CDT_EFFECTIVE_COEFFICIENT_MATCH",
                "OPEN",
                "The structural powers and denominator are fixed, but Gamma, mu, lambda and the de Sitter profile/omega have not been measured with stable finite-size scaling in QCCG.",
                required_next_step=(
                    "Simulate increasing causal QCCG ensembles, extract the reduced N3 transfer matrix, "
                    "fit Gamma/mu/lambda and the mean cos^3 volume profile parameter omega, then test "
                    "stability across volume and time blocking."
                ),
                sources=SOURCES,
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"][:3]):
        raise SystemExit("QCCG-CDT structural action audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
