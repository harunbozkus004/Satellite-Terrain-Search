"""Self-supervised satellite terrain search and change detection."""
import os
import sys

__version__ = "0.1.0"

if sys.platform == "darwin":
    # PyPI wheels of torch and faiss-cpu each bundle their own libomp on macOS.
    # Loading both aborts (OMP Error #15) or segfaults. Allowing the duplicate
    # runtime and keeping FAISS single-threaded (see satterrain.faiss_compat)
    # is the standard workaround; exact search over ~50k vectors is fast anyway.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
