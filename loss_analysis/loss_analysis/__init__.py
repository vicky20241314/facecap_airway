"""Local loss analysis for the facecap_airway_delivery RGB-D pipeline.

This package is **not** part of the reconstruction algorithm. It only observes a
finished run, recomputes the objectives that the source actually defines, and
writes diagnostics. Nothing here is imported by ``head_rgbd`` or ``core``.

Layout
------
objectives   stage/term catalogue transcribed from the delivery source
terms        pure functions computing each term (with exact denominators)
recorder     iteration records with explicit provenance tiers + CSV/JSON writers
adapter      read-only ingestion of an existing run directory
diagnostics  our own MAE/RMSE/P95/coverage on fixed real observations
plots        curves (only from real records) and the Chinese report
cli          `python -m loss_analysis ...`
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
