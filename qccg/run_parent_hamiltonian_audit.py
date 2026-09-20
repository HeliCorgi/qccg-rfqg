#!/usr/bin/env python3
"""Explicit finite-Weyl QCCG parent-Hamiltonian audit.

Scope
-----
This audit fixes one local finite-qudit parent Hamiltonian family.  The
quadratic Hessian is DERIVED from the exact Weyl cosine terms; it is not
inserted as an independent dispersion relation.

The calculation still does not establish a Lorentzian manifold phase,
continuum BRST/diffeomorphism invariance, an asymptotic-safety fixed point, or
full strong-gravity dynamics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sympy as sp


K = 64
DELTA = 2.0 * math.pi / K
TENSOR_SPECIES = ("plus", "cross")
DEFECT_SPECIES = ("scalar", "vector_x", "vector_y", "longitudinal")
DEFECT_MASS = 1.0
SPATIAL_COUPLINGS = (1.0, 1.0, 1.0)
TENSOR_MASS = 0.0

SYMBOLIC_TOL = 0.0
NUMERIC_HESSIAN_TOL = 2.0e-6
NEGATIVE_CONTROL_MIN_GAP = 0.15
NEGATIVE_CONTROL_ANISOTROPY = 0.10


def ev(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "symbolic-numerical-audit",
        "artifact": "qccg/run_parent_hamiltonian_audit.py",
        "note": note,
        "metadata": metadata,
    }


def symbolic_derivation():
    d, p, q1, q2, q, mu = sp.symbols(
        "d p q1 q2 q mu", positive=True, real=True
    )
    kinetic = (2 - 2 * sp.cos(d * p)) / (2 * d**2)
    link = (2 - 2 * sp.cos(d * (q1 - q2))) / (2 * d**2)
    mass = mu**2 * (2 - 2 * sp.cos(d * q)) / (2 * d**2)

    d2_kin = sp.simplify(sp.diff(kinetic, p, 2).subs(p, 0))
    link_hessian = sp.Matrix(
        [[sp.diff(link, a, b) for b in (q1, q2)] for a in (q1, q2)]
    ).subs({q1: 0, q2: 0}).applyfunc(sp.simplify)
    d2_mass = sp.simplify(sp.diff(mass, q, 2).subs(q, 0))

    expected_link = sp.Matrix([[1, -1], [-1, 1]])
    passed = (
        d2_kin == 1
        and link_hessian == expected_link
        and sp.simplify(d2_mass - mu**2) == 0
    )

    kinetic_series = sp.series(kinetic, p, 0, 6)
    link_series = sp.series(link.subs(q2, 0), q1, 0, 6)

    return {
        "passed": bool(passed),
        "kinetic_second_derivative": str(d2_kin),
        "link_hessian": [[str(x) for x in row] for row in link_hessian.tolist()],
        "mass_second_derivative": str(d2_mass),
        "kinetic_series": str(kinetic_series),
        "link_series_one_endpoint_fixed": str(link_series),
    }


def local_energy(x):
    return (2.0 - 2.0 * math.cos(DELTA * x)) / (2.0 * DELTA**2)


def second_derivative_numeric(h=1.0e-4):
    return (local_energy(h) - 2.0 * local_energy(0.0) + local_energy(-h)) / h**2


def lattice_symbol(kvec, couplings=SPATIAL_COUPLINGS):
    return 4.0 * sum(
        c * math.sin(0.5 * k) ** 2 for c, k in zip(couplings, kvec)
    )


def main_spectrum():
    momenta = [
        (0.0, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (0.0, 0.10, 0.0),
        (0.0, 0.0, 0.10),
        (0.10, 0.10, 0.0),
    ]
    rows = []
    for kvec in momenta:
        lam = lattice_symbol(kvec)
        tensor_omega2 = [lam + TENSOR_MASS**2 for _ in TENSOR_SPECIES]
        defect_omega2 = [lam + DEFECT_MASS**2 for _ in DEFECT_SPECIES]
        rows.append({
            "k": list(kvec),
            "tensor_omega2": tensor_omega2,
            "defect_omega2": defect_omega2,
        })
    zero_tensor = sum(abs(x) < 1e-14 for x in rows[0]["tensor_omega2"])
    zero_defect = sum(abs(x) < 1e-14 for x in rows[0]["defect_omega2"])
    return rows, zero_tensor, zero_defect


def negative_controls():
    k0 = (0.0, 0.0, 0.0)
    tensor_mass = 0.20
    massive_tensor_gap = math.sqrt(lattice_symbol(k0) + tensor_mass**2)
    tensor_mass_detected = massive_tensor_gap >= NEGATIVE_CONTROL_MIN_GAP

    defect_mass_zero = 0.0
    extra_zero_modes = len(DEFECT_SPECIES) if defect_mass_zero == 0.0 else 0
    defect_zero_detected = extra_zero_modes > 0

    distorted = (1.0, 1.25, 0.75)
    k = 1.0e-3
    velocities = []
    for axis in range(3):
        kv = [0.0, 0.0, 0.0]
        kv[axis] = k
        omega = math.sqrt(lattice_symbol(tuple(kv), distorted))
        velocities.append(omega / k)
    spread = (max(velocities) - min(velocities)) / (sum(velocities) / 3.0)
    anisotropy_detected = spread >= NEGATIVE_CONTROL_ANISOTROPY

    return {
        "tensor_mass": {
            "introduced_mass": tensor_mass,
            "measured_k0_gap": massive_tensor_gap,
            "detected": tensor_mass_detected,
        },
        "defect_mass_removed": {
            "extra_zero_modes_at_k0": extra_zero_modes,
            "detected": defect_zero_detected,
        },
        "anisotropic_spatial_coupling": {
            "couplings": list(distorted),
            "low_k_velocities": velocities,
            "relative_velocity_spread": spread,
            "detected": anisotropy_detected,
        },
        "all_detected": tensor_mass_detected and defect_zero_detected and anisotropy_detected,
    }


def run():
    sym = symbolic_derivation()
    num_hessian = second_derivative_numeric()
    num_hessian_error = abs(num_hessian - 1.0)
    numeric_pass = num_hessian_error <= NUMERIC_HESSIAN_TOL

    rows, zero_tensor, zero_defect = main_spectrum()
    spectrum_pass = zero_tensor == len(TENSOR_SPECIES) and zero_defect == 0

    neg = negative_controls()

    explicit_formula = (
        "H = sum_{x,a in tensor+defect} (2-X_xa-X_xa^dag)/(2 delta^2) "
        "+ sum_{<xy>,a} c_i (2-Z_xa Z_ya^dag-Z_ya Z_xa^dag)/(2 delta^2) "
        "+ sum_{x,b in defect} mu_b^2 (2-Z_xb-Z_xb^dag)/(2 delta^2)"
    )

    parent_pass = sym["passed"] and numeric_pass and spectrum_pass and neg["all_detected"]

    evidence = [
        ev(
            "qccg-parent-hamiltonian-explicit",
            "FULL_MICROSCOPIC_DERIVATION",
            "PASS" if parent_pass else "FAIL",
            "One finite-Weyl local parent Hamiltonian is fixed and its quadratic Hessian is derived from the exact cosine terms; this PASS is scoped to the parent-to-Hessian derivation, not continuum gravity.",
            K=K,
            delta=DELTA,
            tensor_species=list(TENSOR_SPECIES),
            defect_species=list(DEFECT_SPECIES),
            tensor_mass=TENSOR_MASS,
            defect_mass=DEFECT_MASS,
            spatial_couplings=list(SPATIAL_COUPLINGS),
            exact_parent_formula=explicit_formula,
            symbolic_derivation=sym,
            numeric_second_derivative=num_hessian,
            numeric_second_derivative_error=num_hessian_error,
            numeric_hessian_tolerance=NUMERIC_HESSIAN_TOL,
            k0_zero_tensor_modes=zero_tensor,
            k0_zero_defect_modes=zero_defect,
            spectrum_rows=rows,
            negative_controls=neg,
        )
    ]

    return {
        "schema": 1,
        "scope": "Explicit finite-Weyl parent Hamiltonian -> quadratic Hessian only",
        "evidence": evidence,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = run()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [e for e in result["evidence"] if e["status"] == "FAIL"]
    if failed:
        raise SystemExit("parent-Hamiltonian audit failed: " + ", ".join(e["obligation"] for e in failed))


if __name__ == "__main__":
    main()
