#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PROJECT_DIR = Path.cwd()
DEFAULT_MEAN_VELOCITY_CSV = Path("mean_recoil/mean_recoil_velocity.csv")
DEFAULT_OUTPUT_ROOT_DIR = None
DEFAULT_DT = 1.0
DEFAULT_TIME_OFFSET = 0.0
DEFAULT_DPI = 200

COMPARE_AXES = ("K", "W", "SN")
ANGLE_Y_MIN_DEG = -180.0
ANGLE_Y_MAX_DEG = 180.0
ANGLE_Y_TICKS_DEG = np.arange(-180.0, 181.0, 45.0, dtype=float)

DEFAULT_FONT_SIZE = 20
LEGEND_FONT_SIZE = 14
MARKER_SIZE = 8
LINE_WIDTH = 3.0
PLOT_COLOR_CYCLE = [
    "#6B7280",
    "#DB2777",
    "#0F766E",
    "#0284C7",
    "#7C3AED",
    "#4F46E5",
]
W_MARKER_CYCLE = ["o", "^", "s", "D", "v", "P", "X", "<", ">", "h", "*"]

ANGLE_TIME_SERIES_FILENAME = "recoil_direction_time_series.csv"
RUN_SUMMARY_FILENAME = "analysis_runs.csv"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    ylabel: str
    title_prefix: str
    file_name: str


METRICS = (
    MetricDefinition(
        name="angle_to_x_deg",
        ylabel="angle to x-axis (deg)",
        title_prefix="Mean recoil velocity: angle to x-axis",
        file_name="recoil_angle_to_x_vs_time.png",
    ),
    MetricDefinition(
        name="direction_angle_deg",
        ylabel="direction angle (deg)",
        title_prefix="Mean recoil velocity: direction angle",
        file_name="recoil_direction_vs_time.png",
    ),
)
METRIC_BY_NAME = {metric.name: metric for metric in METRICS}


@dataclass(frozen=True)
class RunCondition:
    k: int
    w: int
    sn: float


@dataclass(frozen=True)
class ProcessingConfig:
    project_dir: Path
    output_root_dir: Path
    input_csv: Path
    dt: float
    time_offset: float
    dpi: int
    compare_targets: tuple[str, ...]


@dataclass
class AngleSummary:
    input_file: str
    time_s: float
    frame_start: int | None
    frame_end: int | None
    valid_vector_count: int
    ablation_x_px: float
    ablation_y_px: float
    source_vector_csv: str
    mean_folded_velocity_x: float
    mean_folded_velocity_y: float
    mean_folded_velocity_magnitude: float
    direction_unit_x: float
    direction_unit_y: float
    angle_to_x_deg: float
    direction_angle_deg: float


@dataclass
class RunMetrics:
    run_dir: Path
    condition: RunCondition
    output_dir: Path
    summary_rows: list[AngleSummary]

    @property
    def k(self) -> int:
        return self.condition.k

    @property
    def w(self) -> int:
        return self.condition.w

    @property
    def sn(self) -> float:
        return self.condition.sn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean-vector angles from the summary CSV exported by compute_mean_recoil.py."
    )
    parser.add_argument("project_dir_positional", nargs="?", type=Path)
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Project directory used by compute_mean_recoil.py (default: current directory)")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="compute_mean_recoil.py summary CSV; relative paths are resolved against project-dir")
    parser.add_argument("--output-root-dir", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT,
                        help="Acquisition interval per frame in seconds; time uses frame A")
    parser.add_argument("--time-offset", type=float, default=DEFAULT_TIME_OFFSET)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--compare-by", choices=("K", "W", "SN", "all"), default="all")
    parser.add_argument("--k-values", type=int, nargs="+", default=None)
    parser.add_argument("--w-values", type=int, nargs="+", default=None)
    parser.add_argument("--sn-values", type=float, nargs="+", default=None)
    args = parser.parse_args()
    if args.project_dir is None:
        args.project_dir = args.project_dir_positional or DEFAULT_PROJECT_DIR
    del args.project_dir_positional
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("--dt must be a positive finite number")
    if not math.isfinite(args.time_offset):
        parser.error("--time-offset must be finite")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def build_processing_config(args: argparse.Namespace) -> ProcessingConfig:
    project_dir = args.project_dir.expanduser().resolve()
    input_csv = (args.input_csv or DEFAULT_MEAN_VELOCITY_CSV).expanduser()
    if not input_csv.is_absolute():
        input_csv = project_dir / input_csv
    input_csv = input_csv.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"compute_mean_recoil.py summary CSV not found: {input_csv}")
    output_root_dir = (
        project_dir / "recoil_direction"
        if args.output_root_dir is None
        else args.output_root_dir.expanduser().resolve()
    )
    return ProcessingConfig(
        project_dir=project_dir, output_root_dir=output_root_dir,
        input_csv=input_csv, dt=args.dt, time_offset=args.time_offset,
        dpi=args.dpi,
        compare_targets=COMPARE_AXES if args.compare_by == "all" else (args.compare_by,),
    )


def load_mean_velocity_runs(config: ProcessingConfig, args: argparse.Namespace) -> list[RunMetrics]:
    """Read already-folded means from compute_mean_recoil.py; never refold or recompute raw vectors."""
    required = {
        "piv_run", "frame_interval", "window_size_px", "snr_threshold", "input_file", "frame_start", "frame_end",
        "ablation_x_px", "ablation_y_px", "valid_vector_count",
        "mean_folded_velocity_x", "mean_folded_velocity_y", "source_vector_csv",
    }
    grouped: dict[tuple[str, RunCondition], list[AngleSummary]] = {}
    seen: set[tuple[str, RunCondition, int, int]] = set()
    origins: dict[tuple[str, RunCondition], tuple[float, float]] = {}
    with config.input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing mean recoil summary columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                condition = RunCondition(int(row["frame_interval"]), int(row["window_size_px"]), float(row["snr_threshold"]))
                if condition.k <= 0 or condition.w <= 0 or not math.isfinite(condition.sn):
                    raise ValueError("Invalid K/W/SN condition")
                if not matches_optional_filters(condition, args.k_values, args.w_values, args.sn_values):
                    continue
                run_name = row["piv_run"]
                if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
                    raise ValueError("run_dir must be a single directory name")
                frame_start, frame_end = int(row["frame_start"]), int(row["frame_end"])
                if frame_end - frame_start != condition.k:
                    raise ValueError("Frame interval does not match K")
                count = int(row["valid_vector_count"])
                if count <= 0:
                    raise ValueError("vector_count must be positive")
                mx, my = float(row["mean_folded_velocity_x"]), float(row["mean_folded_velocity_y"])
                ox, oy = float(row["ablation_x_px"]), float(row["ablation_y_px"])
                if not all(math.isfinite(value) for value in (mx, my, ox, oy)):
                    raise ValueError("Mean components and origin must be finite")
                key = (run_name, condition)
                identity = (*key, frame_start, frame_end)
                if identity in seen:
                    raise ValueError("Duplicate frame pair within one run")
                seen.add(identity)
                if key in origins and origins[key] != (ox, oy):
                    raise ValueError("Mixed fold centers within one run")
                origins[key] = (ox, oy)
                wx, wy, angle, argument = compute_direction_angles(mx, my)
                grouped.setdefault(key, []).append(AngleSummary(
                    input_file=row["input_file"], time_s=config.time_offset + frame_start * config.dt,
                    frame_start=frame_start, frame_end=frame_end, valid_vector_count=count,
                    ablation_x_px=ox, ablation_y_px=oy, source_vector_csv=row["source_vector_csv"],
                    mean_folded_velocity_x=mx, mean_folded_velocity_y=my,
                    mean_folded_velocity_magnitude=math.hypot(mx, my), direction_unit_x=wx, direction_unit_y=wy,
                    angle_to_x_deg=angle, direction_angle_deg=argument,
                ))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid mean recoil summary row {line_number}: {exc}") from exc
    if not grouped:
        raise ValueError("No mean-vector rows remain after applying K/W/SN filters")
    # Validate every row before creating output files.
    runs = []
    output_conditions: dict[str, RunCondition] = {}
    for (run_name, condition), rows in sorted(
        grouped.items(), key=lambda item: (item[0][1].k, item[0][1].w, item[0][1].sn, item[0][0])
    ):
        if run_name in output_conditions and output_conditions[run_name] != condition:
            raise ValueError(f"Conflicting conditions for run_dir: {run_name}")
        output_conditions[run_name] = condition
        runs.append(RunMetrics(
            run_dir=config.project_dir / run_name, condition=condition,
            output_dir=config.output_root_dir / run_name,
            summary_rows=sorted(rows, key=lambda point: (point.frame_start, point.frame_end)),
        ))
    return runs


def time_series_csv_path(output_dir: Path) -> Path:
    return output_dir / ANGLE_TIME_SERIES_FILENAME


def run_inventory_csv_path(output_root_dir: Path) -> Path:
    return output_root_dir / RUN_SUMMARY_FILENAME


def metric_plot_path(output_dir: Path, metric: MetricDefinition) -> Path:
    return output_dir / metric.file_name


def matches_optional_filters(
    condition: RunCondition,
    k_values: list[int] | None,
    w_values: list[int] | None,
    sn_values: list[float] | None,
) -> bool:
    if k_values is not None and condition.k not in k_values:
        return False
    if w_values is not None and condition.w not in w_values:
        return False
    if sn_values is not None and not any(
        math.isclose(condition.sn, value, rel_tol=0.0, abs_tol=1e-9)
        for value in sn_values
    ):
        return False
    return True


def compute_direction_angles(vector_x: float, vector_y: float) -> tuple[float, float, float, float]:
    vector_norm = math.hypot(vector_x, vector_y)
    if vector_norm <= 0.0:
        return math.nan, math.nan, math.nan, math.nan

    unit_x = vector_x / vector_norm
    unit_y = vector_y / vector_norm
    angle_to_x_rad = math.acos(max(-1.0, min(1.0, unit_x)))
    argument_rad = math.atan2(unit_y, unit_x)
    return (
        unit_x,
        unit_y,
        math.degrees(angle_to_x_rad),
        math.degrees(argument_rad),
    )


def metric_values(rows: list[AngleSummary], metric_name: str) -> np.ndarray:
    return np.asarray([getattr(row, metric_name) for row in rows], dtype=float)


def metric_times(rows: list[AngleSummary]) -> np.ndarray:
    return np.asarray([row.time_s for row in rows], dtype=float)


def write_time_series_rows_csv(rows: list[AngleSummary], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_single_run_metric(
    rows: list[AngleSummary],
    metric: MetricDefinition,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    times = metric_times(rows)
    values = metric_values(rows, metric.name)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_prop_cycle(color=PLOT_COLOR_CYCLE)
    ax.plot(
        times,
        values,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=PLOT_COLOR_CYCLE[0],
    )
    ax.set_xlabel("Time (s)", fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(metric.ylabel, fontsize=DEFAULT_FONT_SIZE)
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=DEFAULT_FONT_SIZE)
    ax.set_ylim(ANGLE_Y_MIN_DEG, ANGLE_Y_MAX_DEG)
    ax.set_yticks(ANGLE_Y_TICKS_DEG)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def metric_label(metric_name: str) -> str:
    try:
        return METRIC_BY_NAME[metric_name].ylabel
    except KeyError as exc:
        raise KeyError(f"Unsupported metric: {metric_name}") from exc


def metric_axis_limits(metric_name: str) -> tuple[float, float] | None:
    if metric_name in METRIC_BY_NAME and metric_name.endswith("_deg"):
        return (ANGLE_Y_MIN_DEG, ANGLE_Y_MAX_DEG)
    return None


def format_condition(condition: RunCondition | int, w: int | None = None, sn: float | None = None) -> str:
    if isinstance(condition, RunCondition):
        k_value = condition.k
        w_value = condition.w
        sn_value = condition.sn
    else:
        if w is None or sn is None:
            raise ValueError("w and sn are required when condition is not a RunCondition")
        k_value = condition
        w_value = w
        sn_value = sn
    return f"K={k_value}, W={w_value}, SN={sn_value:.3f}"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def group_key_for_compare(run: RunMetrics, compare_by: str) -> str:
    if compare_by == "K":
        return f"W{run.w}_SN{run.sn:.3f}"
    if compare_by == "W":
        return f"K{run.k}_SN{run.sn:.3f}"
    if compare_by == "SN":
        return f"K{run.k}_W{run.w}"
    raise ValueError(f"Unsupported compare_by: {compare_by}")


def line_label(run: RunMetrics, compare_by: str) -> str:
    if compare_by == "K":
        return f"K={run.k}"
    if compare_by == "W":
        return f"W={run.w}"
    if compare_by == "SN":
        return f"SN={run.sn:.3f}"
    raise ValueError(f"Unsupported compare_by: {compare_by}")


def sort_grouped_runs(grouped_runs: list[RunMetrics], compare_by: str) -> list[RunMetrics]:
    if compare_by == "K":
        return sorted(grouped_runs, key=lambda run: run.k)
    if compare_by == "W":
        return sorted(grouped_runs, key=lambda run: run.w)
    if compare_by == "SN":
        return sorted(grouped_runs, key=lambda run: run.sn)
    raise ValueError(f"Unsupported compare_by: {compare_by}")


def group_runs(runs: list[RunMetrics], compare_by: str) -> dict[str, list[RunMetrics]]:
    """Group runs so one parameter varies while the others stay fixed."""
    groups: dict[str, list[RunMetrics]] = {}
    for run in runs:
        groups.setdefault(group_key_for_compare(run, compare_by), []).append(run)
    return {
        key: sort_grouped_runs(group, compare_by)
        for key, group in groups.items()
        if len(group) >= 2
    }


def build_w_marker_map(w_values: Iterable[int]) -> dict[int, str]:
    return {
        w: W_MARKER_CYCLE[index % len(W_MARKER_CYCLE)]
        for index, w in enumerate(sorted(set(w_values)))
    }


def build_w_color_map(runs: list[RunMetrics]) -> dict[tuple[int, str], str]:
    grouped_labels: dict[int, list[str]] = {}
    for run in runs:
        grouped_labels.setdefault(run.w, []).append(line_label(run, "K"))

    color_map: dict[tuple[int, str], str] = {}
    for w, labels in grouped_labels.items():
        for index, label in enumerate(sorted(set(labels))):
            color_map[(w, label)] = PLOT_COLOR_CYCLE[index % len(PLOT_COLOR_CYCLE)]
    return color_map


def compute_axis_limits(values: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if np.isclose(minimum, maximum):
        margin = 0.5 if np.isclose(minimum, 0.0) else abs(minimum) * 0.05
        return minimum - margin, maximum + margin
    return minimum, maximum


def compute_global_axis_limits(
    runs: list[RunMetrics],
    metric_name: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_times = np.concatenate([metric_times(run.summary_rows) for run in runs])
    all_values = np.concatenate([metric_values(run.summary_rows, metric_name) for run in runs])
    return (
        compute_axis_limits(all_times, (0.0, 1.0)),
        compute_axis_limits(all_values, (ANGLE_Y_MIN_DEG, ANGLE_Y_MAX_DEG)),
    )


def plot_overlay_metric(
    grouped_runs: list[RunMetrics],
    compare_by: str,
    group_name: str,
    metric_name: str,
    output_path: Path,
    dpi: int,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sorted_runs = sort_grouped_runs(grouped_runs, compare_by)
    w_marker_map = build_w_marker_map(run.w for run in sorted_runs)
    w_color_map = build_w_color_map(sorted_runs)

    for run in sorted_runs:
        label = line_label(run, compare_by)
        times = metric_times(run.summary_rows)
        values = metric_values(run.summary_rows, metric_name)
        ax.plot(
            times,
            values,
            color=w_color_map[(run.w, line_label(run, "K"))],
            marker=w_marker_map[run.w],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
        )

    ax.set_xlabel("Time (s)", fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(metric_label(metric_name), fontsize=DEFAULT_FONT_SIZE)
    ax.set_title(f"{metric_label(metric_name)} comparison: {group_name}", fontsize=DEFAULT_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=DEFAULT_FONT_SIZE)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        axis_limits = metric_axis_limits(metric_name)
        if axis_limits is not None:
            ax.set_ylim(*axis_limits)
    if metric_name.endswith("_deg"):
        ax.set_yticks(ANGLE_Y_TICKS_DEG)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_time_series_csv(run: RunMetrics) -> None:
    write_time_series_rows_csv(run.summary_rows, time_series_csv_path(run.output_dir))


def write_run_inventory(runs: list[RunMetrics], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["piv_run", "frame_interval", "window_size_px", "snr_threshold", "summary_count"],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "piv_run": str(run.run_dir),
                    "frame_interval": run.k,
                    "window_size_px": run.w,
                    "snr_threshold": f"{run.sn:.3f}",
                    "summary_count": len(run.summary_rows),
                }
            )


def compare_one_axis(
    runs: list[RunMetrics],
    compare_by: str,
    output_dir: Path,
    metric_name: str,
    dpi: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> int:
    compare_dir = output_dir / {
        "K": "compare_frame_interval",
        "W": "compare_window_size",
        "SN": "compare_snr_threshold",
    }[compare_by]
    compare_dir.mkdir(parents=True, exist_ok=True)

    groups = group_runs(runs, compare_by)
    if not groups:
        print(f"skip: no comparable groups found for compare_by={compare_by}, metric={metric_name}")
        return 0

    figure_count = 0
    for group_name, grouped_runs in groups.items():
        output_path = compare_dir / f"{sanitize_name(metric_name)}_{sanitize_name(group_name)}.png"
        plot_overlay_metric(
            grouped_runs=grouped_runs,
            compare_by=compare_by,
            group_name=group_name,
            metric_name=metric_name,
            output_path=output_path,
            dpi=dpi,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        figure_count += 1
    return figure_count


def export_run_outputs(runs: list[RunMetrics], output_root_dir: Path) -> None:
    write_run_inventory(runs, run_inventory_csv_path(output_root_dir))
    for run in runs:
        write_time_series_csv(run)


def generate_overlay_figures(runs: list[RunMetrics], config: ProcessingConfig) -> int:
    overlay_figure_count = 0
    for metric in METRICS:
        x_limits, y_limits = compute_global_axis_limits(runs, metric.name)
        if metric.name.endswith("_deg"):
            y_limits = (ANGLE_Y_MIN_DEG, ANGLE_Y_MAX_DEG)
        for compare_by in config.compare_targets:
            overlay_figure_count += compare_one_axis(
                runs=runs,
                compare_by=compare_by,
                output_dir=config.output_root_dir,
                metric_name=metric.name,
                dpi=config.dpi,
                x_limits=x_limits,
                y_limits=y_limits,
            )
    return overlay_figure_count


def print_summary(runs: list[RunMetrics], config: ProcessingConfig, overlay_figure_count: int) -> None:
    print(f"Project directory: {config.project_dir}")
    print(f"Input mean summary: {config.input_csv}")
    print(f"Processed run folders: {len(runs)}")
    print(f"Output root directory: {config.output_root_dir}")
    print(f"Run inventory: {run_inventory_csv_path(config.output_root_dir)}")
    print(f"Generated overlay figures: {overlay_figure_count}")
    for run in runs:
        print(
            f"run={format_condition(run.condition)} "
            f"time_series={time_series_csv_path(run.output_dir)}"
        )


def main() -> None:
    args = parse_args()
    config = build_processing_config(args)

    runs = load_mean_velocity_runs(config, args)
    config.output_root_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        for metric in METRICS:
            plot_single_run_metric(
                run.summary_rows, metric,
                f"{metric.title_prefix}: {format_condition(run.condition)}",
                metric_plot_path(run.output_dir, metric), config.dpi,
            )
    export_run_outputs(runs, config.output_root_dir)
    overlay_figure_count = generate_overlay_figures(runs, config)
    print_summary(runs, config, overlay_figure_count)


if __name__ == "__main__":
    main()
