"""facecap_airway_delivery: single-path RGB-D head reconstruction delivery.

One default algorithm (``facecap-inspired-v2-strict``), one entry point and one
final metric mesh. Research modes and evaluation entries are not part of this
package.
"""

from pathlib import Path


__version__ = (
    Path(__file__).resolve().parents[1] / "VERSION"
).read_text(encoding="utf-8").strip()
