#!/usr/bin/env python3
"""Interacting code-subspace scaling toy for QCCG unitarity.

A one-qubit coarse Hilbert space is embedded isometrically into the even-parity
code subspace of two microscopic qubits:
  |0> -> |00>, |1> -> |11>.

The interacting microscopic Hamiltonian
  H_micro = J X1 X2 + Delta (1 - Z1 Z2)/2
preserves the code and exactly intertwines with
  H_coarse = J X.

A single-site perturbation epsilon X1 is a leakage negative control.

This demonstrates a finite interacting isometric scaling architecture, not the
actual graph-changing QCCG physical Hilbert-space limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


I2 = sp.eye(2)
X = sp.Matrix([[0, 1], [1, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])


def kron(A, B):
    return sp.kronecker_product(A, B)


X1 = kron(X, I2)
X2 = kron(I2, X)
Z1 = kron(Z, I2)
Z2 = kron(I2, Z)
I4 = sp.eye(4)

J = sp.Rational(2, 5)
Delta = sp.Integer(3)
epsilon = sp.Rational(1, 7)

C = sp.simplify((I4 - Z1 * Z2) / 2)
Hmicro = sp.simplify(J * X1 * X2 + Delta * C)
Hcoarse = J * X

# Embedding columns are |00>, |11>.
V = sp.Matrix([
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 1],
])


def comm(A, B):
    return sp.simplify(A * B - B * A)


def zero(A):
    return A == sp.zeros(*A.shape)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-interacting-code-scaling",
        "artifact": "qccg/run_interacting_scaling_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    isometry = sp.simplify(V.T * V) == sp.eye(2)
    code_constraint = C * V == sp.zeros(4, 2)
    intertwiner = sp.simplify(Hmicro * V - V * Hcoarse) == sp.zeros(4, 2)
    preserves_constraint = zero(comm(Hmicro, C))

    # The penalty Hamiltonian has zero on the code and positive energy on odd parity.
    c_eigs = C.eigenvals()
    h_eigs = Hmicro.eigenvals()

    # Leakage negative control.
    Hbad = sp.simplify(Hmicro + epsilon * X1)
    bad_comm = comm(Hbad, C)
    leakage = sp.simplify((I4 - V * V.T) * Hbad * V)
    leakage_detected = (not zero(bad_comm)) and (leakage != sp.zeros(4, 2))

    pass_main = isometry and code_constraint and intertwiner and preserves_constraint

    result = {
        "schema": 1,
        "scope": "two-qubit interacting code-subspace scaling toy",
        "evidence": [
            evidence(
                "qccg-interacting-code-intertwiner",
                "INTERACTING_CODE_INTERTWINER_TOY",
                "PASS" if pass_main else "FAIL",
                "The even-parity microscopic code embeds one coarse qubit isometrically, is exactly preserved by an interacting two-body Hamiltonian, and intertwines with the coarse Hamiltonian.",
                isometry=isometry,
                constraint_annihilates_code=code_constraint,
                Hamiltonian_intertwiner=intertwiner,
                Hamiltonian_commutes_with_constraint=preserves_constraint,
                embedding=[[str(x) for x in row] for row in V.tolist()],
                constraint=[[str(x) for x in row] for row in C.tolist()],
                microscopic_H=[[str(x) for x in row] for row in Hmicro.tolist()],
                coarse_H=[[str(x) for x in row] for row in Hcoarse.tolist()],
                constraint_eigenvalues={str(k): int(v) for k, v in c_eigs.items()},
                microscopic_H_eigenvalues={str(k): int(v) for k, v in h_eigs.items()},
            ),
            evidence(
                "qccg-leakage-perturbation-control",
                "SCALING_LEAKAGE_NEGATIVE_CONTROL",
                "PASS" if leakage_detected else "FAIL",
                "A single-site microscopic perturbation fails to commute with the code constraint and produces leakage outside the coarse embedded subspace.",
                epsilon=str(epsilon),
                bad_commutator=[[str(x) for x in row] for row in bad_comm.tolist()],
                leakage=[[str(x) for x in row] for row in leakage.tolist()],
            ),
            evidence(
                "qccg-graph-changing-scaling-map",
                "QCCG_GRAPH_CHANGING_SCALING_MAP",
                "OPEN",
                "No isometric intertwiner has yet been constructed for the actual graph-changing QCCG physical Hilbert space across increasing cellulation size.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("interacting scaling toy failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
