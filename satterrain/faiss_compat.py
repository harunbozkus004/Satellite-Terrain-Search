"""Import FAISS after torch, single-threaded on macOS (see satterrain/__init__.py)."""
import sys

import torch  # noqa: F401  (load torch's libomp first)
import faiss

if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)

__all__ = ["faiss"]
