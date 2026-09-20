# Migration provenance

This repository was split from the QCCG research work that originally lived inside `HeliCorgi/theorygate`.

- Source repository: https://github.com/HeliCorgi/theorygate
- Source branch: `qccg-theorygate-audit`
- Source commit: `33290aa6c4fec7653b3bd6d53b7d3b7fc0f3bbee`
- Source commit message: `Add focused affine CDT kinetic workflow`
- Migration target: `HeliCorgi/qccg-rfqg`
- Migrated research files under `qccg/`: 80

## What was moved

- all QCCG research scripts;
- Lean proofs and toolchain files;
- `qccg/theorygate.yaml` claim/obligation ledger;
- QCCG-specific GitHub Actions workflows;
- literature/source notes.

## What was intentionally not moved

The generic TheoryGate engine, its tests, schema implementation, and unrelated examples/docs remain in the TheoryGate repository. This repository depends on that engine at the pinned source commit in `requirements.txt` for reproducibility.

The new repository's own `LICENSE` was retained.
