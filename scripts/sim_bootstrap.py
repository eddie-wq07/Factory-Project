"""Import this first in every simulator script: it prepares the process before Isaac Sim starts.

* Puts the project's ``src`` and ``scripts`` folders on the import path, so the scripts run without
  installing the package.
* Pins OpenBLAS to one thread. NumPy and SciPy otherwise start a thread pool at import, and Isaac Sim
  forks a helper process during start-up; OpenBLAS's fork handler then joins those threads and
  intermittently crashes or hangs the launch. The IK and planning matrices here are tiny, so a
  single BLAS thread costs nothing.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
