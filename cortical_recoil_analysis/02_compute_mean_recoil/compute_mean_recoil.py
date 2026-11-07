#!/usr/bin/env python3
from __future__ import annotations

"""Export sums and means of velocities reflected around the ablation site."""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


INPUT_GLOB = "piv_results_k*_w*_snr*"
CSV_GLOB = "A*_B*.csv"
RUN_DIR_PATTERN = re.compile(
    r"piv_results_k(?P<frame_interval>\d+)_w(?P<window_size_px>\d+)_snr(?P<snr_threshold>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
FILE_PATTERN = re.compile(
    r"A(?P<frame_start>\d+)_B(?P<frame_end>\d+)(?:_u_val)?\.csv$",
    re.IGNORECASE,
)
COLUMN_ALIASES = {
    "x": ("x",),
    "y": ("y",),
    "u": ("u", "u_val"),
    "v": ("v", "v_val"),
}
OUTPUT_FIELDNAMES = [
    "piv_run",
    "frame_interval",
    "window_size_px",
    "snr_threshold",
    "input_file",
    "frame_start",
    "frame_end",
    "ablation_x_px",
    "ablation_y_px",
    "valid_vector_count",
    "sum_folded_velocity_x",
    "sum_folded_velocity_y",
    "sum_folded_velocity_magnitude",
    "mean_folded_velocity_x",
    "mean_folded_velocity_y",
    "mean_folded_velocity_magnitude",
    "mean_magnitude_from_sum",
    "mean_magnitude_check_error",
    "mean_consistency_check_passed",
    "source_vector_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For every PIV CSV, fold the validated velocities around the ablation site "
            "and export their sum, count, and mean velocity."
        )
    )
    parser.add_argument("project_dir_positional", nargs="?", type=Path)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project directory containing piv_results_k*_w*_snr* folders",
    )
    parser.add_argument("--origin-x", type=float, default=None, help="Fold-center x coordinate")
    parser.add_argument("--origin-y", type=float, default=None, help="Fold-center y coordinate")
    parser.add_argument("--input-glob", default=INPUT_GLOB, help="Run-directory glob")
    parser.add_argument("--csv-glob", default=CSV_GLOB, help="Input CSV glob inside each run")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Combined output CSV. Defaults to "
            "PROJECT_DIR/mean_recoil/mean_recoil_velocity.csv"
        ),
    )
    parser.add_argument("--k-values", type=int, nargs="*", default=None)
    parser.add_argument("--w-values", type=int, nargs="*", default=None)
    parser.add_argument("--sn-values", type=float, nargs="*", default=None)
    args = parser.parse_args()

    if args.project_dir is None:
        args.project_dir = args.project_dir_positional
    del args.project_dir_positional
    if args.project_dir is None:
        parser.error("Specify a project directory positionally or with --project-dir.")

    args.project_dir = args.project_dir.expanduser().resolve()
    if args.output_csv is None:
        args.output_csv = (
            args.project_dir
            / "mean_recoil"
            / "mean_recoil_velocity.csv"
        )
    else:
        args.output_csv = args.output_csv.expanduser().resolve()
    return args


def resolve_origin(origin_x: float | None, origin_y: float | None) -> tuple[float, float]:
    if origin_x is None:
        origin_x = float(input("Enter fold-center x coordinate: ").strip())
    if origin_y is None:
        origin_y = float(input("Enter fold-center y coordinate: ").strip())
    if not math.isfinite(origin_x) or not math.isfinite(origin_y):
        raise ValueError("origin_x and origin_y must be finite numbers.")
    return origin_x, origin_y


def parse_run_condition(run_dir: Path) -> tuple[int, int, float]:
    match = RUN_DIR_PATTERN.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(f"Unexpected run-directory name: {run_dir.name}")
    return int(match.group("frame_interval")), int(match.group("window_size_px")), float(match.group("snr_threshold"))


def parse_frame_numbers(csv_path: Path) -> tuple[int, int]:
    match = FILE_PATTERN.fullmatch(csv_path.name)
    if match is None:
        raise ValueError(f"Unexpected PIV CSV name: {csv_path.name}")
    return int(match.group("frame_start")), int(match.group("frame_end"))


def resolve_column(fieldnames: list[str], canonical_name: str) -> str:
    for alias in COLUMN_ALIASES[canonical_name]:
        if alias in fieldnames:
            return alias
    raise KeyError(
        f"Required column {canonical_name!r} was not found. Available columns: {fieldnames}"
    )


def load_piv_vectors(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    last_decode_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with csv_path.open("r", newline="", encoding=encoding) as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                names = {
                    name: resolve_column(fieldnames, name)
                    for name in COLUMN_ALIASES
                }
                rows = list(reader)
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
    else:
        raise ValueError(f"Could not decode {csv_path}") from last_decode_error

    if not rows:
        raise ValueError(f"No vector rows found in {csv_path}")

    x = np.asarray([float(row[names["x"]]) for row in rows], dtype=float)
    y = np.asarray([float(row[names["y"]]) for row in rows], dtype=float)
    u = np.asarray([float(row[names["u"]]) for row in rows], dtype=float)
    v = np.asarray([float(row[names["v"]]) for row in rows], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    if not np.any(finite):
        raise ValueError(f"No finite x, y, u, v rows found in {csv_path}")
    return x[finite], y[finite], u[finite], v[finite]


def compute_folded_velocity_statistics(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> dict[str, int | float | bool]:
    vector_count = int(u.size)
    if vector_count == 0:
        raise ValueError("At least one valid vector is required.")

    x_reflection = np.where(x - origin_x < 0.0, -1.0, 1.0)
    y_reflection = np.where(y - origin_y < 0.0, -1.0, 1.0)
    p_folded_u = u * x_reflection
    p_folded_v = v * y_reflection

    p_sum_x = float(np.sum(p_folded_u))
    p_sum_y = float(np.sum(p_folded_v))
    p_sum_norm = float(math.hypot(p_sum_x, p_sum_y))

    # Calculate the mean directly from the folded vectors. This is intentionally
    # separate from p_sum / N so the exported identity can verify the calculation.
    p_mean_x = float(np.mean(p_folded_u))
    p_mean_y = float(np.mean(p_folded_v))
    p_mean_norm = float(math.hypot(p_mean_x, p_mean_y))

    expected_mean_norm = p_sum_norm / vector_count
    mean_norm_abs_error = abs(p_mean_norm - expected_mean_norm)
    calculation_check_passed = math.isclose(
        p_mean_norm,
        expected_mean_norm,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    return {
        "valid_vector_count": vector_count,
        "sum_folded_velocity_x": p_sum_x,
        "sum_folded_velocity_y": p_sum_y,
        "sum_folded_velocity_magnitude": p_sum_norm,
        "mean_folded_velocity_x": p_mean_x,
        "mean_folded_velocity_y": p_mean_y,
        "mean_folded_velocity_magnitude": p_mean_norm,
        "mean_magnitude_from_sum": expected_mean_norm,
        "mean_magnitude_check_error": mean_norm_abs_error,
        "mean_consistency_check_passed": calculation_check_passed,
    }


def matches_filters(
    condition: tuple[int, int, float],
    k_values: list[int] | None,
    w_values: list[int] | None,
    sn_values: list[float] | None,
) -> bool:
    k, w, sn = condition
    if k_values is not None and k not in k_values:
        return False
    if w_values is not None and w not in w_values:
        return False
    if sn_values is not None and not any(math.isclose(sn, value) for value in sn_values):
        return False
    return True


def collect_rows(args: argparse.Namespace, origin_x: float, origin_y: float) -> list[dict[str, object]]:
    run_dirs = sorted(
        (
            path
            for path in args.project_dir.glob(args.input_glob)
            if path.is_dir() and RUN_DIR_PATTERN.fullmatch(path.name)
        ),
        key=parse_run_condition,
    )
    run_dirs = [
        path
        for path in run_dirs
        if matches_filters(
            parse_run_condition(path),
            args.k_values,
            args.w_values,
            args.sn_values,
        )
    ]
    if not run_dirs:
        raise FileNotFoundError(
            f"No matching PIV result directories found in {args.project_dir}"
        )

    output_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        k, w, sn = parse_run_condition(run_dir)
        csv_files = sorted(
            (
                path
                for path in run_dir.glob(args.csv_glob)
                if path.is_file() and FILE_PATTERN.fullmatch(path.name)
            ),
            key=parse_frame_numbers,
        )
        if not csv_files:
            print(f"skip: no input CSV files in {run_dir}")
            continue

        for csv_path in csv_files:
            frame_a, frame_b = parse_frame_numbers(csv_path)
            x, y, u, v = load_piv_vectors(csv_path)
            metrics = compute_folded_velocity_statistics(x, y, u, v, origin_x, origin_y)
            output_rows.append(
                {
                    "piv_run": run_dir.name,
                    "frame_interval": k,
                    "window_size_px": w,
                    "snr_threshold": f"{sn:.3f}",
                    "input_file": csv_path.name,
                    "frame_start": frame_a,
                    "frame_end": frame_b,
                    "ablation_x_px": origin_x,
                    "ablation_y_px": origin_y,
                    **metrics,
                    "source_vector_csv": str(csv_path),
                }
            )
    if not output_rows:
        raise FileNotFoundError("No usable PIV CSV files were found.")
    return output_rows


def write_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    origin_x, origin_y = resolve_origin(args.origin_x, args.origin_y)
    rows = collect_rows(args, origin_x, origin_y)
    write_csv(args.output_csv, rows)

    failed_checks = sum(not bool(row["mean_consistency_check_passed"]) for row in rows)
    print(f"Project directory: {args.project_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Processed frames: {len(rows)}")
    print(f"Calculation checks failed: {failed_checks}")
    if failed_checks:
        raise RuntimeError("One or more p-mean verification checks failed.")


if __name__ == "__main__":
    main()
