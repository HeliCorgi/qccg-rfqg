#!/usr/bin/env python3
"""Graph-changing scaling-isometry toy for QCCG.

Coarse sector: one logical qubit.
Refined sector: two microscopic qubits with even-parity physical code
  |0> -> |00>, |1> -> |11>.

A graph-change move operator exchanges the coarse sector and the refined code
without changing the logical state.  The move is Hermitian and unitary on the
physical direct-sum subspace.

A wrong refinement map that sends |1> -> |01> is a leakage negative control.

This is an architecture toy, not the actual infinite QCCG cellulation scaling
map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


# Correct isometric refinement V: C^2 -> C^4.
V = sp.Matrix([
    [1, 0],  # |00>
    [0, 0],  # |01>
    [0, 0],  # |10>
    [0, 1],  # |11>
])

# Refined even-parity projector.
P_ref = V * V.T
I2 = sp.eye(2)
I4 = sp.eye(4)

# Physical direct-sum projector on H_coarse ⊕ H_refined.
P_phys = sp.diag(I2, P_ref)

# Graph-change move M = [[0,V^T],[V,0]].
Z22 = sp.zeros(2)
Z44 = sp.zeros(4)
M = Z22.row_join(V.T).col_join(V.row_join(Z44))

# Wrong refinement map leaks |1> to |01>.
W = sp.Matrix([
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0],
])
Mbad = Z22.row_join(W.T).col_join(W.row_join(Z44))


def zero(A):
    return A == sp.zeros(*A.shape)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-graph-change-scaling-toy",
        "artifact": "qccg/run_graph_change_scaling_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    isometry = V.T * V == I2
    refined_physical = P_ref * V == V
    preserves_phys = (I6 := sp.eye(6))
    leakage_good = sp.simplify((I6 - P_phys) * M * P_phys)
    good_preserves = zero(leakage_good)

    # On physical subspace, M^2 = identity (coarse <-> refined code).
    involution_residual = sp.simplify(P_phys * (M * M - I6) * P_phys)
    involution = zero(involution_residual)
    hermitian = M.H == M

    # Logical state preservation: M maps coarse [psi;0] -> [0;V psi].
    a, b = sp.symbols("a b")
    psi_c = sp.Matrix([a, b])
    state_coarse = psi_c.col_join(sp.zeros(4, 1))
    expected_refined = sp.zeros(2, 1).col_join(V * psi_c)
    logical_preserved = sp.simplify(M * state_coarse - expected_refined) == sp.zeros(6, 1)

    # Negative control.
    bad_leakage = sp.simplify((I6 - P_phys) * Mbad * P_phys)
    bad_detected = not zero(bad_leakage)

    main_pass = isometry and refined_physical and good_preserves and involution and hermitian and logical_preserved

    result = {
        "schema": 1,
        "scope": "finite coarse/refined direct-sum graph-change isometry toy",
        "evidence": [
            evidence(
                "qccg-graph-change-intertwiner-toy",
                "GRAPH_CHANGE_INTERTWINER_TOY",
                "PASS" if main_pass else "FAIL",
                "The coarse/refined graph-change move is an exact Hermitian involution on the physical direct-sum subspace and preserves the logical state under isometric refinement.",
                isometry=isometry,
                refined_code_physical=refined_physical,
                physical_leakage_zero=good_preserves,
                physical_involution=involution,
                move_hermitian=hermitian,
                logical_state_preserved=logical_preserved,
                V=[[str(x) for x in row] for row in V.tolist()],
                move=[[str(x) for x in row] for row in M.tolist()],
            ),
            evidence(
                "qccg-graph-change-wrong-map-control",
                "GRAPH_CHANGE_LEAKAGE_NEGATIVE_CONTROL",
                "PASS" if bad_detected else "FAIL",
                "A wrong refinement map sends one logical basis state into the odd-parity sector and is detected as physical-subspace leakage.",
                W=[[str(x) for x in row] for row in W.tolist()],
                leakage=[[str(x) for x in row] for row in bad_leakage.tolist()],
            ),
            evidence(
                "qccg-actual-graph-changing-scaling",
                "QCCG_GRAPH_CHANGING_SCALING_MAP",
                "OPEN",
                "No compatible family of such intertwiners has yet been constructed across the actual QCCG cellulation graph with local moves, constraints, and increasing system size.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("graph-changing scaling toy failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
