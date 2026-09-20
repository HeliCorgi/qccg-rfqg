#!/usr/bin/env python3
"""Literature-anchor audit for the CDT -> FRG -> Reuter bridge.

This script does NOT promote QCCG by importing external conclusions.
It freezes the published comparison coordinates and caveats that QCCG must
match with its own calculations.

Primary anchors
---------------
1. J. Ambjorn, J. Gizbert-Studnicki, A. Goerlich, D. Nemeth,
   "Is lattice quantum gravity asymptotically safe? Making contact between
   causal dynamical triangulations and the functional renormalization group",
   Phys. Rev. D 110, 126006 (2024),
   DOI: 10.1103/PhysRevD.110.126006.

   The CDT scale-factor variables Gamma and omega are related to continuum
   scale-factor / FRG quantities.  In the round-S4 normalization one obtains

       g_eff^2 = 24*pi*G_k/sqrt(V4)
               = (4/sqrt(6)) * lambda_k * g_k
               ~= 1.633 * lambda_k*g_k.

   A UV lattice path must keep the corresponding dimensionless combination
   finite/nonzero while N4 -> infinity; the paper explicitly stresses that
   current CDT Monte Carlo data allow, but do not prove, a UV fixed point.
   It also warns that the lattice correlation-length scale is not yet
   identified unambiguously with the FRG scale k, so a raw lattice critical
   exponent must not be equated directly with an FRG stability exponent.

2. A. Baldazzi, K. Falls, Y. Kluth, B. Knorr,
   "Robustness of the derivative expansion in asymptotic safety",
   Phys. Rev. D 113, 026005 (2026),
   DOI: 10.1103/hlrm-d4g2, arXiv:2312.03831.

   In the minimal essential scheme at O(partial^6):
       g_N* = 0.364
       g_C3* = 4.490e-7
       theta_1 = 2.225
       theta_2 = -3.850
   with one relevant direction.

These are external anchors only.  QCCG must still measure Gamma/omega/N4 or
an equivalent set of essential observables, locate its own critical point, and
establish the map to the same universality class.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sympy as sp


SOURCE_CDT_FRG = {
    "title": "Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group",
    "journal": "Phys. Rev. D 110, 126006 (2024)",
    "doi": "10.1103/PhysRevD.110.126006",
}
SOURCE_REUTER = {
    "title": "Robustness of the derivative expansion in asymptotic safety",
    "journal": "Phys. Rev. D 113, 026005 (2026)",
    "doi": "10.1103/hlrm-d4g2",
    "arxiv": "2312.03831",
}

lambda_k, g_k = sp.symbols("lambda_k g_k", positive=True, real=True)
analytic_factor = sp.simplify(4 / sp.sqrt(6))
numeric_factor = float(sp.N(analytic_factor, 15))

REUTER_TARGET = {
    "g_N_star": 0.364,
    "g_C3_star": 4.490e-7,
    "theta_relevant": 2.225,
    "theta_irrelevant": -3.850,
    "relevant_directions": 1,
}


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "literature-anchor-symbolic-audit",
        "artifact": "qccg/run_cdt_frg_anchor_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    factor_ok = abs(numeric_factor - 1.632993161855452) < 1e-12
    one_relevant = (
        REUTER_TARGET["theta_relevant"] > 0
        and REUTER_TARGET["theta_irrelevant"] < 0
        and REUTER_TARGET["relevant_directions"] == 1
    )

    result = {
        "schema": 1,
        "scope": "external literature anchors; no QCCG universality promotion",
        "evidence": [
            evidence(
                "cdt-frg-mapping-reference",
                "CDT_FRG_MAPPING_REFERENCE",
                "PASS" if factor_ok else "FAIL",
                "The published CDT/FRG scale-factor comparison supplies a dimensionless matching coordinate proportional to lambda_k*g_k; 4/sqrt(6) is numerically 1.632993...",
                source=SOURCE_CDT_FRG,
                analytic_factor="4/sqrt(6)",
                numeric_factor=numeric_factor,
                relation="g_eff^2 = 24*pi*G_k/sqrt(V4) = (4/sqrt(6))*lambda_k*g_k",
                qccg_import_policy="anchor only; QCCG must independently measure its matching observables",
            ),
            evidence(
                "cdt-uv-scaling-reference",
                "CDT_UV_SCALING_REFERENCE",
                "PASS",
                "The 2024 CDT/FRG comparison provides a finite-size scaling prescription for a putative UV lattice fixed point, but explicitly does not claim that current CDT data prove the fixed point.",
                source=SOURCE_CDT_FRG,
                qccg_required_observables=[
                    "N4",
                    "Gamma",
                    "omega",
                    "critical coupling path",
                    "finite-size scaling",
                ],
                caveat="CDT Monte Carlo evidence allows but does not prove a UV fixed point.",
            ),
            evidence(
                "cdt-frg-exponent-identification-control",
                "CDT_FRG_DIRECT_EXPONENT_IDENTIFICATION_REJECTED",
                "PASS",
                "A raw CDT/lattice critical exponent is not identified directly with an FRG stability exponent because the relation between the lattice correlation-length scale and the FRG coarse-graining scale is not established.",
                source=SOURCE_CDT_FRG,
                consequence="compare universal/essential data only after an explicit scale map or scheme-independent relation is constructed",
            ),
            evidence(
                "reuter-essential-target-reference",
                "REUTER_ESSENTIAL_TARGET_REFERENCE",
                "PASS" if one_relevant else "FAIL",
                "The 2026 minimal-essential derivative expansion supplies a concrete external Reuter-universality target with a unique nontrivial fixed point and one relevant direction at sixth derivative order.",
                source=SOURCE_REUTER,
                target=REUTER_TARGET,
                qccg_import_policy="external comparison target only; not QCCG evidence",
            ),
            evidence(
                "qccg-cdt-frg-coordinates-open",
                "QCCG_CDT_FRG_COORDINATES_MEASURED",
                "OPEN",
                "QCCG has not yet measured stable large-volume Gamma and omega (or an equivalent universal coordinate set) along a controlled critical trajectory, so the published CDT/FRG map cannot yet be numerically applied to QCCG.",
                required_next_step=[
                    "large-volume QCCG causal ensemble",
                    "de Sitter/scale-factor profile fit for omega",
                    "fluctuation or transfer-matrix fit for Gamma",
                    "N4 finite-size scan",
                    "critical trajectory",
                ],
            ),
            evidence(
                "qccg-reuter-essential-match-open",
                "QCCG_REUTER_ESSENTIAL_MATCH",
                "OPEN",
                "No QCCG essential critical-exponent spectrum or fixed-point coordinate map has yet been shown to match the Reuter target.",
                target=REUTER_TARGET,
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"][:4]):
        raise SystemExit("CDT/FRG literature-anchor audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
