"""Kaggle-only availability-faithful B3 versus compact neural comparison.

This runner intentionally has no Test prediction or submission code.  It only
creates historical replay OOF artifacts and model/preprocessor checkpoints.
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import _attach_delta
from scripts.run_hydrological_trajectory import INNER_ORIGINS, OUTER_ORIGINS, _fold
from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.availability import build_replay_observation_view
from src.ml_features import SOURCE_CORE_COLUMNS, SOURCE_HYDRO_HISTORY_COLUMNS, build_hydro_gap_safe_feature_matrix, build_sampled_training_rows, horizon_rebalance_weights
from src.neural_sequence import (
    ArrayNormalizer,
    build_b3_feature_maps,
    build_b3_matrix,
    build_temporal_tensor,
    deterministic_horizon_cap,
    make_model,
)
from src.regional_context import HYDRO_COLUMNS, attach_regional_features
from src.validation import build_test_mask_template

SEEDS = (20260908, 20260909)
STABILITY_SEED = 20260910
SPANS = (6, 12)
MAX_EPOCHS = 30
PER_HORIZON_CAP = 36_000
LEARNABILITY_ROWS = 2_048


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_ids(values: pd.Series) -> str:
    return hashlib.sha256(values.astype(str).str.cat(sep="\n").encode("utf-8")).hexdigest()


def _memory() -> dict[str, float | str]:
    try:
        import psutil
        process = psutil.Process()
        return {"rss_gib": process.memory_info().rss / 1024**3, "available_gib": psutil.virtual_memory().available / 1024**3}
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return {"note": f"{type(exc).__name__}: {exc}"}


def _emit(log, value: object) -> None:
    text = json.dumps(value, sort_keys=True, default=str)
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


def _labels(rows: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    return _attach_delta(rows, labels).astype(np.float32)


def _b0_matrix(ledger: pd.DataFrame, source: pd.DataFrame, structural: pd.DataFrame, regional: pd.DataFrame) -> pd.DataFrame:
    base = build_hydro_gap_safe_feature_matrix(ledger, source, structural)
    result = pd.concat([base.reset_index(drop=True), attach_regional_features(ledger.sample_id, regional).reset_index(drop=True)], axis=1)
    if result.columns.duplicated().any() or np.isinf(result.to_numpy(dtype=np.float32)).any():
        raise AssertionError("availability-faithful B0 feature matrix is invalid")
    return result.astype(np.float32)


def _metadata(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.loc[:, ["sample_id", "source_date", "last_observed_date", "h", "lat", "lon"]].copy()
    source = pd.to_datetime(result.source_date).dt.to_period("M").astype("int64")
    anchor = pd.to_datetime(result.last_observed_date).dt.to_period("M").astype("int64")
    result["anchor_age_months"] = (source - anchor).astype(np.int16)
    result["geo5"] = (
        (np.floor(result.lat.astype(float) / 5.0) * 5.0).astype(int).astype(str)
        + ":" + (np.floor(result.lon.astype(float) / 5.0) * 5.0).astype(int).astype(str)
    )
    result["calendar_block"] = pd.to_datetime(result.source_date).dt.to_period("M").astype(str)
    return result


def _oof(ledger: pd.DataFrame, labels: pd.DataFrame, delta: np.ndarray, *, origin: str, model: str, seed: int | None, span: int | None, epoch: int | None) -> pd.DataFrame:
    target = labels.set_index("sample_id").loc[ledger.sample_id, "target"].to_numpy(dtype=np.float64)
    prediction = ledger.last_observed_TWS.to_numpy(dtype=np.float64) + np.asarray(delta, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError("OOF prediction is non-finite")
    result = _metadata(ledger)
    result["target"] = target
    result["prediction"] = prediction
    result["residual"] = prediction - target
    result["origin"] = origin
    result["model"] = model
    result["seed"] = seed if seed is not None else -1
    result["span"] = span if span is not None else 0
    result["epoch"] = epoch if epoch is not None else 0
    return result


def _rmse(oof: pd.DataFrame) -> tuple[float, dict[str, float]]:
    out: dict[str, float] = {}
    for horizon, group in oof.groupby("h", sort=True):
        out[str(int(horizon))] = float(np.sqrt(np.mean(np.square(group.residual.to_numpy(dtype=np.float64)))))
    return float(np.sqrt(np.mean(np.square(oof.residual.to_numpy(dtype=np.float64))))), out


def _chosen_epoch(rows: list[dict[str, object]], frozen_epoch: int | None) -> dict[str, object]:
    """Select an inner best epoch, or use a previously frozen epoch exactly."""
    if frozen_epoch is None:
        return min(rows, key=lambda row: (float(row["valid_raw_rmse"]), int(row["epoch"])))
    matches = [row for row in rows if int(row["epoch"]) == frozen_epoch]
    if len(matches) != 1:
        raise ValueError(f"frozen epoch {frozen_epoch} is absent from trained curve")
    return matches[0]


def _torch_setup(seed: int):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return torch, device


def _weighted_loss(torch, prediction, target, weight):
    loss = (weight * (prediction - target).square()).sum() / weight.sum().clamp_min(1e-8)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite weighted loss")
    return loss


def _predict(torch, model, static: np.ndarray, sequence: np.ndarray | None, *, device, batch_size: int = 2048) -> np.ndarray:
    model.eval(); result: list[np.ndarray] = []
    with torch.no_grad():
        for first in range(0, len(static), batch_size):
            last = min(len(static), first + batch_size)
            xs = torch.from_numpy(static[first:last]).to(device)
            xq = None if sequence is None else torch.from_numpy(sequence[first:last]).to(device)
            prediction = model(xs, xq).detach().cpu().numpy()
            result.append(prediction.astype(np.float32, copy=False))
    output = np.concatenate(result)
    if not np.isfinite(output).all():
        raise FloatingPointError("non-finite neural prediction")
    return output


def _learnability(torch, architecture: str, static: np.ndarray, sequence: np.ndarray | None, target: np.ndarray, weight: np.ndarray, *, seed: int, device) -> dict[str, float]:
    count = min(LEARNABILITY_ROWS, len(static))
    model = make_model(architecture, static_dim=static.shape[1], sequence_dim=None if sequence is None else sequence.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    x = torch.from_numpy(static[:count]).to(device); q = None if sequence is None else torch.from_numpy(sequence[:count]).to(device)
    y = torch.from_numpy(target[:count]).to(device); w = torch.from_numpy(weight[:count]).to(device)
    losses: list[float] = []
    model.train()
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_loss(torch, model(x, q), y, w); loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite gradient in learnability check")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); losses.append(float(loss.detach().cpu()))
    if not losses[-1] < losses[0]:
        raise RuntimeError({"learnability_failed": architecture, "initial": losses[0], "final": losses[-1], "seed": seed})
    return {"initial_loss": losses[0], "final_loss": losses[-1], "rows": count}


def _fit_neural(
    *, output: Path, architecture: str, origin: str, seed: int, span: int | None,
    static_train: np.ndarray, static_valid: np.ndarray, sequence_train: np.ndarray | None,
    sequence_valid: np.ndarray | None, y_train: np.ndarray, y_valid: np.ndarray,
    weight_train: np.ndarray, ledger: pd.DataFrame, labels: pd.DataFrame,
    static_columns: list[str], sequence_columns: tuple[str, ...] | None, epochs: int = MAX_EPOCHS,
    frozen_epoch: int | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    torch, device = _torch_setup(seed)
    run_name = f"{origin}_{architecture}_s{span or 0}_seed{seed}"
    run_dir = output / "neural" / run_name; run_dir.mkdir(parents=True, exist_ok=False)
    static_norm = ArrayNormalizer.fit(static_train); static_norm.save(run_dir / "static_normalizer.npz", columns=static_columns)
    train_static = static_norm.transform(static_train); valid_static = static_norm.transform(static_valid)
    if sequence_train is not None:
        sequence_norm = ArrayNormalizer.fit(sequence_train); sequence_norm.save(run_dir / "sequence_normalizer.npz", columns=sequence_columns or ())
        train_sequence = sequence_norm.transform(sequence_train); valid_sequence = sequence_norm.transform(sequence_valid)
    else:
        train_sequence = valid_sequence = None
    # Persistence proof: loading the saved arrays yields exactly the same normalized tensors.
    loaded_static, loaded_columns = ArrayNormalizer.load(run_dir / "static_normalizer.npz")
    if tuple(static_columns) != loaded_columns or not np.array_equal(train_static, loaded_static.transform(static_train)):
        raise AssertionError("static preprocessor reload differs")
    if train_sequence is not None:
        loaded_sequence, loaded_channels = ArrayNormalizer.load(run_dir / "sequence_normalizer.npz")
        if tuple(sequence_columns or ()) != loaded_channels or not np.array_equal(train_sequence, loaded_sequence.transform(sequence_train)):
            raise AssertionError("sequence preprocessor reload differs")
    if not np.isfinite(train_static).all() or not np.isfinite(valid_static).all() or (train_sequence is not None and (not np.isfinite(train_sequence).all() or not np.isfinite(valid_sequence).all())):
        raise FloatingPointError("normalization did not make tensors finite")
    learnability = _learnability(torch, architecture, train_static, train_sequence, y_train, weight_train, seed=seed, device=device)
    model = make_model(architecture, static_dim=train_static.shape[1], sequence_dim=None if train_sequence is None else train_sequence.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    rng = np.random.default_rng(seed + 10_000)
    epoch_rows: list[dict[str, object]] = []
    checkpoint_by_epoch: dict[int, Path] = {}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); shuffled = rng.permutation(len(train_static)); total, denominator = 0.0, 0
        for first in range(0, len(shuffled), 1024):
            idx = shuffled[first:first + 1024]
            xs = torch.from_numpy(train_static[idx]).to(device)
            xq = None if train_sequence is None else torch.from_numpy(train_sequence[idx]).to(device)
            y = torch.from_numpy(y_train[idx]).to(device); w = torch.from_numpy(weight_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True); loss = _weighted_loss(torch, model(xs, xq), y, w); loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite full-fit gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach().cpu()) * len(idx); denominator += len(idx)
        delta = _predict(torch, model, valid_static, valid_sequence, device=device)
        record_oof = _oof(ledger, labels, delta, origin=origin, model=architecture, seed=seed, span=span, epoch=epoch)
        raw, by_h = _rmse(record_oof)
        record = {"origin": origin, "architecture": architecture, "span": span or 0, "seed": seed, "epoch": epoch,
                  "train_weighted_mse": total / max(denominator, 1), "valid_raw_rmse": raw, "by_h": by_h,
                  "elapsed_seconds": time.perf_counter() - started, "device": str(device), "learnability": learnability}
        epoch_rows.append(record)
        checkpoint = run_dir / f"epoch_{epoch:02d}.pt"
        torch.save({"architecture": architecture, "static_dim": train_static.shape[1], "sequence_dim": None if train_sequence is None else train_sequence.shape[2], "epoch": epoch, "seed": seed, "state_dict": model.state_dict()}, checkpoint)
        checkpoint_by_epoch[epoch] = checkpoint
    curve = pd.DataFrame([{**row, "by_h": json.dumps(row["by_h"], sort_keys=True)} for row in epoch_rows])
    curve.to_csv(run_dir / "curve.csv", index=False)
    best_epoch = _chosen_epoch(epoch_rows, frozen_epoch)
    checkpoint_path = checkpoint_by_epoch[int(best_epoch["epoch"])]
    checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state["state_dict"])
    best_delta = _predict(torch, model, valid_static, valid_sequence, device=device)
    best_oof = _oof(ledger, labels, best_delta, origin=origin, model=architecture, seed=seed, span=span, epoch=int(best_epoch["epoch"]))
    best_oof.to_csv(run_dir / "best_epoch_oof.csv.gz", index=False, compression="gzip")
    # Model reload proof uses the saved state and persisted normalizers, not the live object.
    restored = make_model(architecture, static_dim=train_static.shape[1], sequence_dim=None if train_sequence is None else train_sequence.shape[2]).to(device)
    restored.load_state_dict(checkpoint_state["state_dict"])
    restored_delta = _predict(torch, restored, loaded_static.transform(static_valid), None if sequence_valid is None else loaded_sequence.transform(sequence_valid), device=device)
    if not np.array_equal(best_delta, restored_delta):
        raise AssertionError("checkpoint/preprocessor reload prediction differs")
    _json(run_dir / "manifest.json", {"status": "completed", "run": run_name, "device": str(device), "learnability": learnability,
                                        "best_epoch": best_epoch, "static_columns": static_columns, "sequence_columns": sequence_columns,
                                        "checkpoint": checkpoint_path.name, "reload_prediction_equal": True, "elapsed_seconds": time.perf_counter() - started})
    return epoch_rows, best_oof


def _fit_baseline(lgb, *, output: Path, candidate: str, origin: str, train_x: pd.DataFrame, valid_x: pd.DataFrame, y_train: np.ndarray, weights: np.ndarray, ledger: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    params = dict(DEFAULT_PARAMS, num_leaves=63, min_data_in_leaf=1000, num_threads=min(4, os.cpu_count() or 1), seed=20260908,
                  feature_fraction_seed=20260908, bagging_seed=20260908, data_random_seed=20260908)
    started = time.perf_counter(); model = lgb.train(params, lgb.Dataset(train_x, label=y_train, weight=weights, feature_name=list(train_x.columns)), num_boost_round=98, callbacks=[lgb.log_evaluation(0)])
    delta = model.predict(valid_x).astype(np.float32); oof = _oof(ledger, labels, delta, origin=origin, model=candidate, seed=None, span=None, epoch=98)
    raw, by_h = _rmse(oof); path = output / "baseline"; path.mkdir(exist_ok=True)
    model.save_model(str(path / f"{origin}_{candidate}.txt")); oof.to_csv(path / f"{origin}_{candidate}_oof.csv.gz", index=False, compression="gzip")
    return {"origin": origin, "candidate": candidate, "raw_rmse": raw, "by_h": by_h, "feature_count": int(train_x.shape[1]), "elapsed_seconds": time.perf_counter() - started,
            "train_ids_hash": _hash_ids(pd.Series(train_x.index.astype(str))), "valid_ids_hash": _hash_ids(ledger.sample_id)}


def _selection(epoch_rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(epoch_rows).copy()
    # Keep the declared static-MLP sentinel robust if a future caller supplies
    # `None`; pandas would otherwise drop that grouping key.
    frame["span"] = frame["span"].fillna(0).astype(int)
    expected = len(INNER_ORIGINS) * len(SEEDS)
    grouped = frame.groupby(["architecture", "span", "epoch"], sort=True).agg(raw_rmse=("valid_raw_rmse", "mean"), runs=("valid_raw_rmse", "size")).reset_index()
    grouped = grouped.loc[grouped.runs == expected].sort_values(["raw_rmse", "architecture", "span", "epoch"], kind="mergesort")
    if grouped.empty:
        raise AssertionError("no complete inner neural recipe exists for selection")
    chosen = grouped.iloc[0].to_dict()
    return {"selection_metric": "equally weighted mean raw RMSE over two inner origins and two seeds", "chosen": {k: (int(v) if k in {"span", "epoch", "runs"} else (float(v) if k == "raw_rmse" else v)) for k, v in chosen.items()}, "all_complete_recipes": grouped.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the predeclared Kaggle-only neural sequence comparison.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--phase", choices=("inner", "outer", "stability", "ablation"), required=True)
    parser.add_argument("--selection", type=Path, help="inner selection.json required outside inner phase")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--origins", help="comma-separated subset of the predeclared origins; intended only to resume a failed origin")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("refusing to run from dirty checkout")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    selection: dict[str, object] | None = None
    if args.phase != "inner":
        if args.selection is None or not args.selection.is_file():
            raise ValueError("--selection from a completed inner run is required")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))["chosen"]
    started = time.perf_counter()
    configured_origins = INNER_ORIGINS if args.phase in {"inner", "stability", "ablation"} else OUTER_ORIGINS
    if args.origins:
        origins = tuple(part.strip() for part in args.origins.split(",") if part.strip())
        if not origins or any(origin not in configured_origins for origin in origins):
            raise ValueError(f"--origins must be a nonempty subset of {configured_origins}")
    else:
        origins = configured_origins
    manifest = {"status": "running", "commit": head, "phase": args.phase, "started_at": datetime.now(timezone.utc).isoformat(), "no_test_predictions": True,
                "seeds": list(SEEDS), "stability_seed": STABILITY_SEED, "spans": list(SPANS), "max_epochs": args.epochs, "per_horizon_cap": PER_HORIZON_CAP,
                "platform": platform.platform(), "origins": list(origins), "memory_start": _memory()}
    _json(args.output_dir / "manifest.json", manifest)
    try:
        import lightgbm as lgb
        train_cols = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
        test_cols = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]
        train = pd.read_csv(args.data_dir / "Train.csv", usecols=train_cols)
        test = pd.read_csv(args.data_dir / "Test.csv", usecols=test_cols)
        template = build_test_mask_template(test); del test
        source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        labels = train.loc[:, ["sample_id", "target"]].copy()
        baseline_results: list[dict[str, object]] = []; epoch_rows: list[dict[str, object]] = []; all_best_oof: list[pd.DataFrame] = []
        with (args.output_dir / "console.log").open("w", encoding="utf-8") as log:
            _emit(log, {"phase": "start", "origins": origins, "memory": _memory()})
            for origin in origins:
                fold, role = _fold(train, template, origin)
                view = build_replay_observation_view(source, structural, ledger=fold.ledger, first_source_month=fold.spec.first_source_month, last_source_month=fold.spec.last_source_month)
                sampled = build_sampled_training_rows(view.structural, view.source.loc[:, SOURCE_CORE_COLUMNS], max_target_month=pd.Period(origin, freq="M") - 1, seed=20260908)
                train_rows = sampled.rows; y_full = _labels(train_rows, labels); weights_full = np.asarray(horizon_rebalance_weights(train_rows.h), dtype=np.float32)
                maps = build_b3_feature_maps(view.source)
                valid_b3 = build_b3_matrix(fold.ledger, view.source, view.structural, maps)
                _emit(log, {"phase": "features_ready", "origin": origin, "role": role, "train_rows": len(train_rows), "valid_rows": len(fold.ledger),
                            "available_prefix_rows": view.prefix_rows, "scheduled_rows": view.scheduled_rows, "withheld_window_rows": view.withheld_window_rows, "memory": _memory()})
                if args.phase in {"inner", "outer"}:
                    train_b0 = _b0_matrix(train_rows, view.source, view.structural, maps.regional)
                    valid_b0 = _b0_matrix(fold.ledger, view.source, view.structural, maps.regional)
                    result = _fit_baseline(lgb, output=args.output_dir, candidate="B0", origin=origin, train_x=train_b0, valid_x=valid_b0, y_train=y_full, weights=weights_full, ledger=fold.ledger, labels=labels)
                    baseline_results.append(result); _emit(log, {"phase": "baseline_complete", **result})
                    # Building B3 for late origins is several GiB.  Release B0 before
                    # materializing it so a failed origin can be resumed within Kaggle's cgroup limit.
                    del train_b0, valid_b0
                    gc.collect()
                    train_b3 = build_b3_matrix(train_rows, view.source, view.structural, maps)
                    result = _fit_baseline(lgb, output=args.output_dir, candidate="B3", origin=origin, train_x=train_b3, valid_x=valid_b3, y_train=y_full, weights=weights_full, ledger=fold.ledger, labels=labels)
                    baseline_results.append(result); _emit(log, {"phase": "baseline_complete", **result})
                    del train_b3
                    gc.collect()
                cap_rows = deterministic_horizon_cap(train_rows, per_horizon=PER_HORIZON_CAP)
                cap_y = _labels(cap_rows, labels); cap_w = np.asarray(horizon_rebalance_weights(cap_rows.h), dtype=np.float32)
                static_train = build_b3_matrix(cap_rows, view.source, view.structural, maps).to_numpy(dtype=np.float32)
                static_valid = valid_b3.to_numpy(dtype=np.float32)
                valid_target = labels.set_index("sample_id").loc[fold.ledger.sample_id, "target"].to_numpy(dtype=np.float32)
                if args.phase == "inner":
                    configurations = [("mlp", None, seed) for seed in SEEDS] + [(architecture, span, seed) for architecture in ("gru", "tcn") for span in SPANS for seed in SEEDS]
                elif args.phase == "outer":
                    configurations = [(str(selection["architecture"]), int(selection["span"]) or None, seed) for seed in SEEDS]
                elif args.phase == "stability":
                    configurations = [(str(selection["architecture"]), int(selection["span"]) or None, STABILITY_SEED)]
                else:
                    architecture, span = str(selection["architecture"]), int(selection["span"]) or None
                    if architecture == "mlp":
                        raise RuntimeError("history ablation is structurally inapplicable to selected static MLP")
                    configurations = [(architecture, span, seed) for seed in SEEDS]
                cached_sequences: dict[int, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
                for architecture, span, seed in configurations:
                    if span is not None and span not in cached_sequences:
                        train_tensor = build_temporal_tensor(view.source, cap_rows.sample_id, span=span)
                        valid_tensor = build_temporal_tensor(view.source, fold.ledger.sample_id, span=span)
                        cached_sequences[span] = (train_tensor.values, valid_tensor.values, train_tensor.channels)
                    seq_train, seq_valid, sequence_columns = cached_sequences[span] if span is not None else (None, None, None)
                    if args.phase == "ablation":
                        seq_train = seq_train.copy(); seq_valid = seq_valid.copy()
                        # Preserve current source-month channels; hide all older raw sequence slots.
                        seq_train[:, :-1, :] = np.nan; seq_valid[:, :-1, :] = np.nan
                    curves, best = _fit_neural(output=args.output_dir, architecture=architecture, origin=origin, seed=seed, span=span,
                                                static_train=static_train, static_valid=static_valid, sequence_train=seq_train, sequence_valid=seq_valid,
                                                y_train=cap_y, y_valid=valid_target, weight_train=cap_w, ledger=fold.ledger, labels=labels,
                                                static_columns=valid_b3.columns.tolist(), sequence_columns=sequence_columns, epochs=args.epochs if args.phase == "inner" else int(selection["epoch"]),
                                                frozen_epoch=None if args.phase == "inner" else int(selection["epoch"]))
                    epoch_rows.extend(curves); all_best_oof.append(best); _emit(log, {"phase": "neural_complete", "origin": origin, "architecture": architecture, "span": span, "seed": seed, "best": min(curves, key=lambda x: x["valid_raw_rmse"]), "memory": _memory()})
                del cached_sequences, static_train, static_valid, valid_b3, maps
        if baseline_results:
            _json(args.output_dir / "baseline_metrics.json", baseline_results)
        if epoch_rows:
            pd.DataFrame([{**row, "by_h": json.dumps(row["by_h"], sort_keys=True), "learnability": json.dumps(row["learnability"], sort_keys=True)} for row in epoch_rows]).to_csv(args.output_dir / "neural_epoch_metrics.csv", index=False)
            pd.concat(all_best_oof, ignore_index=True).to_csv(args.output_dir / "neural_best_oof.csv.gz", index=False, compression="gzip")
        if args.phase == "inner":
            selected = _selection(epoch_rows); _json(args.output_dir / "selection.json", selected)
        manifest.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "memory_end": _memory(),
                         "checksums": {str(p.relative_to(args.output_dir)): _hash_file(p) for p in args.output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"}})
        _json(args.output_dir / "manifest.json", manifest)
        package = Path(shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir))
        manifest["package"] = {"path": str(package), "sha256": _hash_file(package), "bytes": package.stat().st_size}; _json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}", "memory_end": _memory()})
        _json(args.output_dir / "manifest.json", manifest); raise


if __name__ == "__main__":
    main()
