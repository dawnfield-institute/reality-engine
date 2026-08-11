"""Reality Engine — package root.

**The engine is `src.v3`.** Import it as `from src.v3.engine import ...`.

This package deliberately holds nothing but `v3/`. It is kept as the import root rather
than promoting `v3` to the top level because v3 addresses itself absolutely — 31 modules
and 72 test references use `src.v3.*`. Moving it would mean rewriting every one of those
for no behavioural gain.

Earlier generations are in `archive/`, preserved rather than deleted:

  archive/v1/   the top-level layer packages (substrate, conservation, dynamics,
                scales, emergence, core, ...) that `.spec/architecture.spec.md`
                originally documented — January 2026
  archive/v2/   this package's previous contents, Reality Engine 2.0.0a1 — February 2026

Dawn Field Institute.
"""

__version__ = "3.0.0a1"
