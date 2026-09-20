#!/usr/bin/env python3
"""Lorentz-universality audit for the QCCG parent pilot.

This explicitly distinguishes:
- irrelevant lattice k^4 anisotropy;
- marginal two-derivative velocity anisotropy;
- a model-level common causal propagator that removes the latter at the bare
  quadratic point;
- the still-open interacting RG stability of that common cone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BASE_COUPLINGS = (1.0, 1.0, 1.0)
SPECIES_SPEED_SQUARED = {
    "plus": 1.0,
    "cross": 1.0,
    "matter_probe": 1.0,
}
BROKEN_COUPLINGS = (1.0, 1.25, 0.75)
BROKEN_MATTER_SPEED_SQUARED = 1.30
LOW_K = 1.0e-4
ISOTROPY_TOL = 1.0e-7
COMMON_CONE_TOL = 1.0e-12


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "rg-analytic-audit",
        "artifact": "qccg/run_lorentz_audit.py",
        "note": note,
        "metadata": metadata,
    }


def lattice_symbol(kvec, couplings):
    return 4.0 * sum(c * math.sin(0.5 * k) ** 2 for c, k in zip(couplings, kvec))


def velocities(couplings):
    vals = []
    for axis in range(3):
        kv = [0.0, 0.0, 0.0]
        kv[axis] = LOW_K
        vals.append(math.sqrt(lattice_symbol(tuple(kv), couplings)) / LOW_K)
    return vals


def run():
    base_v = velocities(BASE_COUPLINGS)
    base_spread = (max(base_v) - min(base_v)) / (sum(base_v) / 3.0)
    bare_isotropic = base_spread <= ISOTROPY_TOL

    species_v = {name: math.sqrt(c2) for name, c2 in SPECIES_SPEED_SQUARED.items()}
    species_spread = (max(species_v.values()) - min(species_v.values()))
    common_cone = species_spread <= COMMON_CONE_TOL

    # Tree-level Wilson scaling:
    # two-derivative velocity differences are dimension-4 / marginal in 3+1D;
    # lattice four-derivative corrections carry two extra derivatives.
    scaling_rows = []
    for b in (2, 4, 8, 16):
        scaling_rows.append({
            "b": b,
            "two_derivative_velocity_anisotropy_factor": 1.0,
            "four_derivative_lattice_factor": b ** -2,
        })
    lattice_irrelevant = scaling_rows[-1]["four_derivative_lattice_factor"] < scaling_rows[0]["four_derivative_lattice_factor"]

    broken_v = velocities(BROKEN_COUPLINGS)
    broken_spread = (max(broken_v) - min(broken_v)) / (sum(broken_v) / 3.0)
    # Under tree-level rescaling the relative velocity spread stays constant.
    broken_after_b16 = broken_spread
    marginal_problem_detected = broken_after_b16 > 0.1

    broken_species = dict(SPECIES_SPEED_SQUARED)
    broken_species["matter_probe"] = BROKEN_MATTER_SPEED_SQUARED
    broken_species_v = {name: math.sqrt(c2) for name, c2 in broken_species.items()}
    broken_species_spread = max(broken_species_v.values()) - min(broken_species_v.values())
    species_problem_detected = broken_species_spread > 0.1

    result = {
        "schema": 1,
        "scope": "QCCG bare quadratic common-cone / tree-level RG diagnostic",
        "diagnosis": {
            "generic_tree_level_result": (
                "Higher-derivative lattice Lorentz violation is irrelevant, but "
                "two-derivative direction/species velocity differences are marginal "
                "and do not flow away by canonical scaling alone."
            ),
            "repair": (
                "The selected parent pilot uses one common propagation coefficient "
                "for all audited species and cubic-symmetric spatial links.  This "
                "removes marginal velocity splitting at the bare quadratic point. "
                "Interacting RG protection remains unproved."
            ),
        },
        "evidence": [
            evidence(
                "qccg-common-causal-propagator-choice",
                "COMMON_CAUSAL_PROPAGATOR",
                "MODEL_CHOICE",
                "The pilot declares one shared causal propagation coefficient for tensor and probe-matter species; this is an explicit microscopic design principle, not an RG derivation.",
                species_speed_squared=SPECIES_SPEED_SQUARED,
            ),
            evidence(
                "qccg-bare-spatial-isotropy",
                "BARE_SPATIAL_ISOTROPY",
                "PASS" if bare_isotropic and common_cone else "FAIL",
                "The selected quadratic parent has cubic-symmetric spatial velocity and a common bare cone for the audited species.",
                directional_velocities=base_v,
                directional_relative_spread=base_spread,
                species_velocities=species_v,
                species_absolute_spread=species_spread,
                isotropy_tolerance=ISOTROPY_TOL,
                common_cone_tolerance=COMMON_CONE_TOL,
            ),
            evidence(
                "qccg-lattice-lv-irrelevant",
                "LATTICE_LV_IRRELEVANT",
                "PASS" if lattice_irrelevant else "FAIL",
                "Four-derivative lattice corrections scale as b^-2 relative to the two-derivative term.",
                scaling=scaling_rows,
            ),
            evidence(
                "qccg-marginal-lv-stability",
                "MARGINAL_LV_STABILITY",
                "OPEN",
                "No interacting RG calculation yet proves that the common causal cone is stable against all allowed marginal/relevant Lorentz-violating operators.",
                broken_directional_couplings=BROKEN_COUPLINGS,
                broken_directional_velocities=broken_v,
                broken_relative_spread=broken_spread,
                broken_relative_spread_after_b16=broken_after_b16,
                marginal_problem_detected=marginal_problem_detected,
                broken_species_speed_squared=broken_species,
                broken_species_velocities=broken_species_v,
                species_problem_detected=species_problem_detected,
            ),
        ],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = run()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
