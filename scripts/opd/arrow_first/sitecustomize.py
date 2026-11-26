"""Job-scoped workaround for the import-order SIGSEGV reproduced on h004.

Only put this directory on PYTHONPATH for an affected OPD job. Do not install it
in site-packages: it intentionally controls startup only for that job's Python
processes, including inherited subprocesses and Ray workers.
"""

import os

if os.environ.get("OPD_ARROW_FIRST") == "1":
    try:
        import pyarrow  # noqa: F401
    except Exception as error:
        # Python normally reports and ignores errors from sitecustomize. Do not
        # continue into the known bad import order if this workaround failed.
        raise SystemExit(f"OPD PyArrow startup preload failed: {error!r}") from error
    OPD_ARROW_PRELOADED = True
