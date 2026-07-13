"""Compatibility wrapper for the released L9 calibration workflow.

The manuscript now uses the unified calibration implementation in
``run_l9_calibration.py``.  Keeping this wrapper avoids two competing L9
implementations, obsolete parameter names, and mixed output directories.
"""

from __future__ import annotations

from run_l9_calibration import main


if __name__ == "__main__":
    main()
