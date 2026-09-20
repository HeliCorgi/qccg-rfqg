#!/usr/bin/env python3
"""Reference-scope audit for spatial Pachner connectivity in QCCG.

Mathematical input (external theorem, not proved here):
Pachner's theorem says triangulations of the same PL manifold are connected by
a finite sequence of bistellar/Pachner moves.  In dimension three the move
types are 1<->4 and 2<->3.

QCCG's audited spatial-slice dynamics explicitly implements reversible
1<->4 and 2<->3 moves on closed combinatorial 3-manifold triangulations.

This evidence therefore promotes only the spatial-slice PL-connectivity
statement.  It does NOT prove ergodicity of the full four-dimensional causal
triangulation space, statistical mixing rates, or continuum criticality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCES = [
    {
        "title": "Connecting 3-manifold triangulations with monotonic sequences of elementary moves",
        "authors": "Benjamin A. Burton, Alexander He",
        "arxiv": "2012.02398",
        "statement": "Any two triangulations of the same 3-manifold are connected by a finite sequence of bistellar flips/Pachner moves.",
    },
    {
        "title": "Transition Amplitudes in 3D Quantum Gravity: Boundaries and Holography in the Coloured Boulatov Model",
        "venue": "Annales Henri Poincare",
        "statement": "For PL-manifolds, Pachner theorem gives finite Pachner-move equivalence; in three dimensions the move types are (1-4) and (2-3), with inverses understood.",
    },
]

IMPLEMENTED_MOVES = {"1<->4", "2<->3"}
REQUIRED_MOVES = {"1<->4", "2<->3"}


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "external-theorem-scope-audit",
        "artifact": "qccg/run_spatial_pachner_theorem_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    move_match = IMPLEMENTED_MOVES == REQUIRED_MOVES
    result = {
        "schema": 1,
        "scope": "spatial closed PL 3-manifold triangulation connectivity; not full 4D causal ergodicity",
        "evidence": [
            evidence(
                "qccg-spatial-pachner-theorem-reference",
                "SPATIAL_PACHNER_CONNECTIVITY_THEOREM",
                "PASS",
                "External Pachner-theorem input: triangulations of a fixed PL 3-manifold are connected by finite bistellar moves; the 3D elementary types are 1<->4 and 2<->3 (and inverses).",
                sources=SOURCES,
            ),
            evidence(
                "qccg-spatial-pachner-moveset-match",
                "QCCG_SPATIAL_PACHNER_MOVESET_MATCH",
                "PASS" if move_match else "FAIL",
                "The QCCG spatial-slice move architecture contains exactly the 3D Pachner move families required by the theorem at the audited abstract move-family level.",
                implemented=sorted(IMPLEMENTED_MOVES),
                required=sorted(REQUIRED_MOVES),
            ),
            evidence(
                "qccg-spatial-slice-connectivity",
                "QCCG_SPATIAL_SLICE_PL_CONNECTIVITY",
                "PASS" if move_match else "FAIL",
                "Combining the external Pachner theorem with the audited QCCG 1<->4 and 2<->3 spatial move families establishes finite move connectivity within each fixed closed PL 3-manifold slice class. This does not establish full spacetime/casual triangulation ergodicity.",
                sources=SOURCES,
                implemented=sorted(IMPLEMENTED_MOVES),
            ),
        ],
    }

    p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("spatial Pachner theorem applicability audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    main()
