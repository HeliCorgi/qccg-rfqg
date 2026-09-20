#!/usr/bin/env python3
"""Refinement-path coherence toy for QCCG graph-changing scaling.

A local refinement isometry copies the logical basis:
    V |0> = |00>,  V |1> = |11>.

From one coarse qubit to three refined qubits there are two refinement paths:
refine the left child or refine the right child.  On the physical repetition
code both paths must define the same embedding, otherwise the continuum state
depends on arbitrary cellulation-refinement history.

The correct copy isometry is exactly coherent.  A child-dependent phase in one
refinement path is a negative control and is detected.

This remains a finite code toy; it does not prove coherence for the full QCCG
Pachner/cellulation category.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


I2 = sp.eye(2)
X = sp.Matrix([[0, 1], [1, 0]])

V = sp.Matrix([
    [1, 0],  # 00
    [0, 0],  # 01
    [0, 0],  # 10
    [0, 1],  # 11
])

# A deliberately bad right-child refinement: |1> acquires a minus sign.
V_bad = sp.Matrix([
    [1, 0],
    [0, 0],
    [0, 0],
    [0, -1],
])


def kron(A, B):
    return sp.kronecker_product(A, B)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-refinement-coherence-toy",
        "artifact": "qccg/run_refinement_coherence_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    # Both maps: C^2 -> C^8.
    left_path = sp.simplify(kron(V, I2) * V)
    right_path = sp.simplify(kron(I2, V) * V)
    coherent = left_path == right_path

    isometry = sp.simplify(left_path.T * left_path) == sp.eye(2)

    # Logical X intertwines with X^⊗3.
    X3 = kron(kron(X, X), X)
    intertwines = sp.simplify(X3 * left_path - left_path * X) == sp.zeros(8, 2)

    # Bad right-child convention.
    right_bad = sp.simplify(kron(I2, V_bad) * V)
    bad_residual = sp.simplify(left_path - right_bad)
    bad_detected = bad_residual != sp.zeros(8, 2)

    result = {
        "schema": 1,
        "scope": "finite repetition-code local refinement coherence toy",
        "evidence": [
            evidence(
                "qccg-refinement-path-coherence",
                "REFINEMENT_PATH_COHERENCE_TOY",
                "PASS" if coherent and isometry and intertwines else "FAIL",
                "The two one-to-three refinement paths agree exactly on the physical code, remain isometric, and intertwine the logical X with the three-site logical operator.",
                left_path=[[str(x) for x in row] for row in left_path.tolist()],
                right_path=[[str(x) for x in row] for row in right_path.tolist()],
                coherent=coherent,
                isometry=isometry,
                logical_intertwiner=intertwines,
            ),
            evidence(
                "qccg-refinement-phase-negative-control",
                "REFINEMENT_COHERENCE_NEGATIVE_CONTROL",
                "PASS" if bad_detected else "FAIL",
                "A child-dependent refinement phase makes the final physical embedding path-dependent and is detected.",
                bad_right_path=[[str(x) for x in row] for row in right_bad.tolist()],
                residual=[[str(x) for x in row] for row in bad_residual.tolist()],
            ),
            evidence(
                "qccg-full-cellulation-coherence-open",
                "QCCG_CELLULATION_REFINEMENT_COHERENCE",
                "OPEN",
                "No coherent family has yet been verified over the full QCCG Pachner/cellulation move groupoid, including nontrivial loops of local moves.",
                next_step=(
                    "Assign local physical isometries to all allowed QCCG moves and audit loop/path "
                    "independence on a generating set of Pachner relations before claiming a "
                    "cellulation-independent physical scaling map."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("refinement coherence toy failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
