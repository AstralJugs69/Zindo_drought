"""Benchmark one fixed cached B3 inference workload at a chosen thread count.

The benchmark intentionally uses an existing B3/98 booster and one deterministic
446-column matrix cache.  It performs no fitting, reads no Test data, and writes
only the cache/result requested by the caller.  Run each thread count in a fresh
process with matching OMP/BLAS environment variables to avoid nested
oversubscription.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

try:  # Linux VM only; keep module importable from the Windows checkout.
    import resource
except ImportError:  # pragma: no cover
    resource = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads not in {12, 24, 48}:
        raise ValueError("benchmark threads must be one of 12, 24, 48")
    if args.rows < 1 or args.repeats < 1:
        raise ValueError("rows and repeats must be positive")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.cache_dir / f"b3_inference_matrix_{args.rows}x446.npy"
    if not matrix_path.exists():
        rng = np.random.default_rng(20260910)
        # The first 446 B3 feature names are available in the saved model; this
        # finite, fixed matrix is enough to compare traversal throughput.
        matrix = rng.normal(0.0, 1.0, size=(args.rows, 446)).astype(np.float32)
        np.save(matrix_path, matrix, allow_pickle=False)
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.shape != (args.rows, 446) or matrix.dtype != np.dtype("float32"):
        raise AssertionError("cached benchmark matrix schema changed")

    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(args.model))
    warmup = booster.predict(matrix[: min(2_000, len(matrix))], num_threads=args.threads)
    if not np.isfinite(warmup).all():
        raise AssertionError("warmup inference returned non-finite values")
    timings: list[float] = []
    checksums: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        prediction = booster.predict(matrix, num_threads=args.threads)
        timings.append(time.perf_counter() - started)
        checksums.append(float(np.sum(prediction, dtype=np.float64)))
    if max(checksums) - min(checksums) > 1e-7:
        raise AssertionError("repeated predictions are not deterministic")
    payload = {
        "status": "completed",
        "workload": "fixed_cached_b3_iter98_inference",
        "threads": args.threads,
        "rows": args.rows,
        "features": 446,
        "repeats": args.repeats,
        "warmup_rows": min(2_000, len(matrix)),
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "model_path": str(args.model),
        "model_sha256": sha256_file(args.model),
        "seconds": timings,
        "median_seconds": float(np.median(timings)),
        "rows_per_second_median": float(args.rows / np.median(timings)),
        "prediction_checksum": checksums[0],
        "max_rss_gib": (
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
            if resource is not None else None
        ),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
