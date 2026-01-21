#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Iterable


def _expand_inputs(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for item in inputs:
        paths.append(item)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def _collect_avg_csv(roots: list[str]) -> list[str]:
    candidates = _expand_inputs(roots)
    collected: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for filename in files:
                    if filename.endswith("_results_avg.csv"):
                        path = os.path.normpath(os.path.join(root, filename))
                        if path not in seen:
                            seen.add(path)
                            collected.append(path)
        elif os.path.isfile(item):
            if os.path.basename(item).endswith("_results_avg.csv"):
                path = os.path.normpath(item)
                if path not in seen:
                    seen.add(path)
                    collected.append(path)
        else:
            raise ValueError(f"Input path does not exist: {item}")
    print(f"Found {len(collected)} *_results_avg.csv files.")
    return collected


def _to_float(value: str, *, context: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Failed to parse float for {context}: {value!r}") from exc


def _iter_rows(paths: Iterable[str]) -> Iterable[tuple[str, list[str], dict[str, str]]]:
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV (no header): {path}")
            fieldnames = list(reader.fieldnames)
            for row in reader:
                yield path, fieldnames, row


def _default_out_path(first_input: str) -> str:
    first = os.path.abspath(first_input)
    if os.path.isdir(first):
        return os.path.join(first, "eval_results_avg_by_dataset.csv")
    first_dir = os.path.dirname(first)
    return os.path.join(first_dir, "eval_results_avg_by_dataset.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate *_results_avg.csv files across datasets/subsets by grouping on LoRA config "
            "columns and computing arithmetic mean for metric_eval_*_mean columns."
        )
    )
    parser.add_argument(
        "--r",
        nargs="+",
        required=True,
        help="One or more root paths to search recursively for *_results_avg.csv files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path (default: <root>/eval_results_avg_by_dataset.csv using the first "
            "input root)."
        ),
    )
    parser.add_argument(
        "--key-cols",
        default="base_model,init_lora_weights,extra,r,lora_alpha",
        help="Comma-separated columns used to group rows (default: %(default)s).",
    )
    parser.add_argument(
        "--metric-prefix",
        default="metric_eval_",
        help="Metric column prefix to aggregate (default: %(default)s).",
    )
    parser.add_argument(
        "--metric-suffix",
        default="_mean",
        help="Metric column suffix to aggregate (default: %(default)s).",
    )
    args = parser.parse_args()

    key_columns = [c.strip() for c in args.key_cols.split(",") if c.strip()]
    if not key_columns:
        raise ValueError("No key columns provided.")

    paths = _collect_avg_csv(args.r)
    if not paths:
        raise ValueError("No *_results_avg.csv files found under the given root path(s).")

    grouped: dict[tuple[str, ...], list[float]] = {}
    for path, fieldnames, row in _iter_rows(paths):
        missing = [c for c in key_columns if c not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        row_key = tuple((row.get(col) or "").strip() for col in key_columns)
        metrics = [
            c
            for c in fieldnames
            if c.startswith(args.metric_prefix) and c.endswith(args.metric_suffix)
        ]
        for metric_col in metrics:
            value = (row.get(metric_col) or "").strip()
            if value == "":
                continue
            grouped.setdefault(row_key, []).append(
                _to_float(value, context=f"{metric_col} ({path})")
            )

    if not grouped:
        raise ValueError("No rows found in input CSV(s).")

    out_rows: list[dict[str, str]] = []
    for key, values in grouped.items():
        out: dict[str, str] = {col: val for col, val in zip(key_columns, key)}
        mean = statistics.fmean(values)
        out["metric_eval_mean_avg"] = f"{mean:.10g}"
        out["metric_eval_mean_n"] = str(len(values))
        out_rows.append(out)

    out_rows.sort(
        key=lambda row: _to_float(row.get("metric_eval_mean_avg", "0"), context="metric_eval_mean_avg"),
        reverse=True,
    )

    fieldnames: list[str] = []
    fieldnames.extend(key_columns)
    fieldnames.append("metric_eval_mean_avg")
    fieldnames.append("metric_eval_mean_n")

    out_path = args.out or _default_out_path(args.r[0])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote aggregated results to {out_path}")


if __name__ == "__main__":
    main()
