#!/usr/bin/env python3
"""Linearized spin-2 constraint-target audit for QCCG.

Checks, for several nonzero momentum directions:
- three momentum constraints plus one Hamiltonian constraint have rank 4 in
  the 12-dimensional symmetric-tensor phase space;
- their linear Poisson-bracket matrix vanishes (first-class at this scope);
- the symmetric-tensor TT projector is idempotent and has rank 2.

This is a continuum/linearized TARGET audit.  It does not prove that the
finite-Weyl parent dynamically generates these constraints, nor that the full
nonlinear hypersurface-deformation algebra closes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


SQ2 = sp.sqrt(2)
# Orthonormal basis of real symmetric 3x3 tensors.
BASIS = []
for i in range(3):
    M = sp.zeros(3)
    M[i, i] = 1
    BASIS.append(M)
for i, j in ((0, 1), (0, 2), (1, 2)):
    M = sp.zeros(3)
    M[i, j] = 1 / SQ2
    M[j, i] = 1 / SQ2
    BASIS.append(M)

MOMENTA = (
    sp.Matrix([1, 0, 0]),
    sp.Matrix([1, 1, 0]),
    sp.Matrix([1, 1, 1]),
    sp.Matrix([2, 1, 3]),
)


def inner(A, B):
    return sp.simplify(sp.trace(A.T * B))


def tensor_from_coeff(v):
    out = sp.zeros(3)
    for c, E in zip(v, BASIS):
        out += c * E
    return out


def tt_project(H, k):
    k2 = sp.simplify((k.T * k)[0])
    P = sp.eye(3) - (k * k.T) / k2
    PHP = P * H * P
    trp = sp.simplify(sp.trace(PHP))
    return sp.simplify(PHP - sp.Rational(1, 2) * P * trp)


def tt_matrix(k):
    cols = []
    for E in BASIS:
        T = tt_project(E, k)
        cols.append(sp.Matrix([inner(F, T) for F in BASIS]))
    return sp.Matrix.hstack(*cols)


def constraint_matrix(k):
    # Phase-space coordinates y=(h_6, pi_6).
    k2 = sp.simplify((k.T * k)[0])

    # Hamiltonian constraint C0 = k_i k_j h_ij - k^2 tr(h).
    c0_h = []
    for E in BASIS:
        c0_h.append(sp.simplify((k.T * E * k)[0] - k2 * sp.trace(E)))

    rows = [sp.Matrix([[*c0_h, *([0] * 6)]])]

    # Momentum constraints Ci = k_j pi^{ij}.
    for i in range(3):
        coeff = []
        for E in BASIS:
            coeff.append(sp.simplify((E * k)[i]))
        rows.append(sp.Matrix([[*([0] * 6), *coeff]]))

    return sp.Matrix.vstack(*rows)


def symplectic_matrix():
    I = sp.eye(6)
    Z = sp.zeros(6)
    return Z.row_join(I).col_join((-I).row_join(Z))


def audit_one(k):
    C = constraint_matrix(k)
    J = symplectic_matrix()
    PB = sp.simplify(C * J * C.T)

    Ptt = tt_matrix(k)
    idempotence = (Ptt * Ptt - Ptt).applyfunc(sp.simplify)

    # Verify projected basis tensors are transverse/traceless generically via matrix action.
    projected_checks = []
    for E in BASIS:
        T = tt_project(E, k)
        projected_checks.append({
            "trace": str(sp.simplify(sp.trace(T))),
            "transverse": [str(sp.simplify(x)) for x in (T * k)],
        })

    return {
        "k": [str(x) for x in k],
        "constraint_rank": int(C.rank()),
        "poisson_zero": PB == sp.zeros(4),
        "tt_rank": int(Ptt.rank()),
        "tt_idempotent": idempotence == sp.zeros(6),
        "tt_trace": str(sp.simplify(sp.trace(Ptt))),
        "projected_checks": projected_checks,
    }


def ev(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-linearized-constraint-audit",
        "artifact": "qccg/run_linearized_constraint_audit.py",
        "note": note,
        "metadata": metadata,
    }


def run():
    rows = [audit_one(k) for k in MOMENTA]
    constraints_pass = all(r["constraint_rank"] == 4 and r["poisson_zero"] for r in rows)
    tt_pass = all(r["tt_rank"] == 2 and r["tt_idempotent"] and r["tt_trace"] == "2" for r in rows)
    dof_count = 12 - 2 * 4
    config_dof = dof_count // 2
    return {
        "schema": 1,
        "scope": "linearized continuum symmetric-tensor constraint target",
        "evidence": [
            ev(
                "qccg-linearized-first-class-constraints",
                "LINEARIZED_CONSTRAINT_ALGEBRA",
                "PASS" if constraints_pass else "FAIL",
                "The target linearized Hamiltonian/momentum constraints have rank four and vanishing Poisson-bracket matrix for the audited nonzero momentum directions.",
                phase_space_dimension=12,
                first_class_constraints=4,
                physical_phase_space_dimension=dof_count,
                physical_configuration_dof=config_dof,
                rows=rows,
            ),
            ev(
                "qccg-tt-projector-rank-two",
                "TT_PROJECTOR_RANK2",
                "PASS" if tt_pass else "FAIL",
                "The target symmetric-tensor TT projector is idempotent with rank and trace two for the audited nonzero momentum directions.",
                rows=rows,
            ),
            ev(
                "qccg-nonlinear-constraint-closure",
                "NONLINEAR_CONSTRAINT_CLOSURE",
                "OPEN",
                "No microscopic or continuum interacting calculation yet establishes anomaly-free nonlinear hypersurface-deformation/BRST closure.",
            ),
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = run()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [e for e in result["evidence"] if e["status"] == "FAIL"]
    if failed:
        raise SystemExit("linearized constraint audit failed: " + ", ".join(e["obligation"] for e in failed))


if __name__ == "__main__":
    main()
