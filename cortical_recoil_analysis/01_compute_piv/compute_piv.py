# Compute whole-field PIV velocities (pixel/s) without image masks.
# python3 compute_piv.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re
import tkinter as tk
from tkinter import filedialog, simpledialog

import imageio.v3 as iio
import numpy as np
from openpiv import filters, pyprocess, validation
from scipy.io import savemat


# =========================
# User settings
# =========================
# Initial directory for the project-selection dialog; override with --project-dir.
PROJECT_DIR = Path.cwd()
IMAGE_DIR = PROJECT_DIR / "raw_images"
START_FRAME = 0
END_FRAME = 24
FRAME_INTERVAL_S = 1
PIXEL_SIZE_UM = 0.0681

# Sweep parameters
# ここに試したい値を並べるだけで、全組み合わせを自動で計算します
FRAME_INTERVAL_VALUES = [1]
WINDOW_SIZE_PX_VALUES = [16]
SNR_THRESHOLD_VALUES = [1.025]

# PIV parameter rules derived from W
OVERLAP_RATIO = 0.5
SEARCH_AREA_SIZE_OFFSET = 0

# replace invalid vectors
INTERPOLATE_INVALID_VECTORS = False
OUTLIER_MAX_ITER = 3
OUTLIER_KERNEL_SIZE = 2


# =========================
# Helpers
# =========================
def extract_index(path: Path) -> Optional[int]:
    """Extract trailing number before extension, e.g. image1300.tif -> 1300."""
    m = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if m is None:
        return None
    return int(m.group(1))


def list_images_in_range(raw_dir: Path, start_idx: int, end_idx: int) -> List[Tuple[int, Path]]:
    files = []
    for p in sorted(raw_dir.glob("*.tif")):
        if p.name.startswith("._"):
            continue
        idx = extract_index(p)
        if idx is None:
            continue
        if start_idx <= idx <= end_idx:
            files.append((idx, p))
    return files


def load_gray_image(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return np.asarray(img, dtype=np.float32)


def validate_vectors(
    u: np.ndarray,
    v: np.ndarray,
    sig2noise: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    invalid_mask = validation.sig2noise_val(sig2noise, threshold=threshold)
    u1 = u.copy()
    v1 = v.copy()
    u1[invalid_mask] = np.nan
    v1[invalid_mask] = np.nan
    return u1, v1, invalid_mask


def save_vector_csv(
    csv_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    drop_nan: bool = False,
    header: str = "x,y,u,v",
) -> None:
    data = np.column_stack((x.flatten(), y.flatten(), u.flatten(), v.flatten()))

    if drop_nan:
        data = data[~np.isnan(data).any(axis=1)]

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def build_param_dir_name(k: int, window_size: int, s2n_threshold: float) -> str:
    return f"piv_results_k{k}_w{window_size}_snr{s2n_threshold:.3f}"


def compute_overlap(window_size: int) -> int:
    overlap = int(round(window_size * OVERLAP_RATIO))
    if overlap >= window_size:
        raise ValueError("OVERLAP_RATIO is too large. overlap must be smaller than window_size.")
    return overlap


def compute_search_area_size(window_size: int) -> int:
    search_area_size = int(window_size + SEARCH_AREA_SIZE_OFFSET)
    if search_area_size < window_size:
        raise ValueError("SEARCH_AREA_SIZE_OFFSET makes search_area_size smaller than window_size.")
    return search_area_size


def validate_parameter_values(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")


def parse_numeric_list(text: str, cast, name: str) -> List[float]:
    values = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(cast(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid value in {name}: {stripped}") from exc
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def build_runtime_settings() -> Tuple[Path, Path, List[int], List[int], List[float]]:
    parser = argparse.ArgumentParser(
        description="Run OpenPIV batch processing with runtime-selectable directory and parameters."
    )
    parser.add_argument(
        "project_dir_positional",
        nargs="?",
        type=Path,
        default=None,
        help="Project directory containing raw_images",
    )
    parser.add_argument("--project-dir", type=Path, help="Project directory containing raw_images")
    parser.add_argument("--k", help="Comma-separated K values, e.g. 1,2,3")
    parser.add_argument("--w", help="Comma-separated window sizes, e.g. 8,16,32")
    parser.add_argument("--sn", help="Comma-separated signal-to-noise thresholds, e.g. 1.00,1.05")
    args = parser.parse_args()
    if args.project_dir is None:
        args.project_dir = args.project_dir_positional
    del args.project_dir_positional

    project_dir = args.project_dir
    k_values = FRAME_INTERVAL_VALUES.copy()
    window_size_values = WINDOW_SIZE_PX_VALUES.copy()
    s2n_threshold_values = SNR_THRESHOLD_VALUES.copy()

    root = None
    if project_dir is None or args.k is None or args.w is None or args.sn is None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

    if project_dir is None:
        selected_dir = filedialog.askdirectory(
            title="解析に使用するプロジェクトディレクトリを選択してください",
            initialdir=str(PROJECT_DIR),
        )
        if not selected_dir:
            raise RuntimeError("Project directory selection was cancelled.")
        project_dir = Path(selected_dir)

    if args.k is not None:
        k_values = parse_numeric_list(args.k, int, "K")
    elif root is not None:
        k_text = simpledialog.askstring(
            "K values",
            "K をカンマ区切りで入力してください",
            initialvalue=",".join(map(str, FRAME_INTERVAL_VALUES)),
            parent=root,
        )
        if k_text is None:
            raise RuntimeError("K input was cancelled.")
        k_values = parse_numeric_list(k_text, int, "K")

    if args.w is not None:
        window_size_values = parse_numeric_list(args.w, int, "W")
    elif root is not None:
        w_text = simpledialog.askstring(
            "W values",
            "W をカンマ区切りで入力してください",
            initialvalue=",".join(map(str, WINDOW_SIZE_PX_VALUES)),
            parent=root,
        )
        if w_text is None:
            raise RuntimeError("W input was cancelled.")
        window_size_values = parse_numeric_list(w_text, int, "W")

    if args.sn is not None:
        s2n_threshold_values = parse_numeric_list(args.sn, float, "SN")
    elif root is not None:
        sn_text = simpledialog.askstring(
            "SN values",
            "SN をカンマ区切りで入力してください",
            initialvalue=",".join(f"{value:.3f}" for value in SNR_THRESHOLD_VALUES),
            parent=root,
        )
        if sn_text is None:
            raise RuntimeError("SN input was cancelled.")
        s2n_threshold_values = parse_numeric_list(sn_text, float, "SN")

    if root is not None:
        root.destroy()

    raw_dir = project_dir / "raw_images"
    return project_dir, raw_dir, k_values, window_size_values, s2n_threshold_values


def process_single_combination(
    *,
    project_dir: Path,
    k: int,
    window_size: int,
    s2n_threshold: float,
    idx_to_path: Dict[int, Path],
    image_cache: Dict[Path, np.ndarray],
) -> int:
    overlap = compute_overlap(window_size)
    search_area_size = compute_search_area_size(window_size)
    out_dir = project_dir / build_param_dir_name(k, window_size, s2n_threshold)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_count = 0

    print(
        f"=== Start: K={k}, W={window_size}, SN={s2n_threshold:.3f}, "
        f"overlap={overlap}, search_area={search_area_size} ==="
    )

    for frame_a in range(START_FRAME, END_FRAME - k + 1):
        frame_b = frame_a + k

        if frame_a not in idx_to_path or frame_b not in idx_to_path:
            continue

        path_a = idx_to_path[frame_a]
        path_b = idx_to_path[frame_b]

        if path_a not in image_cache:
            image_cache[path_a] = load_gray_image(path_a)
        if path_b not in image_cache:
            image_cache[path_b] = load_gray_image(path_b)

        img_a = image_cache[path_a]
        img_b = image_cache[path_b]

        if img_a.shape != img_b.shape:
            raise ValueError(f"Shape mismatch: {path_a.name} vs {path_b.name}")

        frame_a_int = img_a.astype(np.int32)
        frame_b_int = img_b.astype(np.int32)
        dt_seconds = k * FRAME_INTERVAL_S

        u_raw, v_raw, sig2noise = pyprocess.extended_search_area_piv(
            frame_a_int,
            frame_b_int,
            window_size=window_size,
            overlap=overlap,
            dt=dt_seconds,
            search_area_size=search_area_size,
            sig2noise_method="peak2mean",
        )

        x, y = pyprocess.get_coordinates(
            image_size=frame_a_int.shape,
            search_area_size=search_area_size,
            overlap=overlap,
        )

        u_val, v_val, invalid_mask = validate_vectors(
            u_raw.copy(), v_raw.copy(), sig2noise, s2n_threshold
        )

        if INTERPOLATE_INVALID_VECTORS:
            u_out, v_out = filters.replace_outliers(
                u_val,
                v_val,
                invalid_mask,
                method="localmean",
                max_iter=OUTLIER_MAX_ITER,
                kernel_size=OUTLIER_KERNEL_SIZE,
            )
        else:
            u_out, v_out = u_val, v_val

        out_name = f"A{frame_a:04d}_B{frame_b:04d}.mat"
        out_path = out_dir / out_name

        savemat(
            out_path,
            {
                "velocity_x_raw": u_raw,
                "velocity_y_raw": v_raw,
                "velocity_x_validated": u_val,
                "velocity_y_validated": v_val,
                "velocity_x_final": u_out,
                "velocity_y_final": v_out,
                "x": x,
                "y": y,
                "frame_start": frame_a,
                "frame_end": frame_b,
                "pair_interval_s": dt_seconds,
                "pixel_size_um": PIXEL_SIZE_UM,
                "frame_interval": k,
                "window_size_px": window_size,
                "overlap": overlap,
                "search_area_size": search_area_size,
                "snr_threshold": s2n_threshold,
                "source_image_start": str(path_a),
                "source_image_end": str(path_b),
                "used_mask": False,
            },
        )

        save_vector_csv(
            csv_path=out_dir / f"A{frame_a:04d}_B{frame_b:04d}.csv",
            x=x,
            y=y,
            u=u_val,
            v=v_val,
            drop_nan=True,
            header="x,y,u,v",
        )

        pair_count += 1
        print(f"[K={k}, W={window_size}, SN={s2n_threshold:.3f}] saved: {out_path.name}")

    print(f"=== Done: K={k}, W={window_size}, SN={s2n_threshold:.3f} -> {pair_count} pairs ===")
    return pair_count


# =========================
# Main
# =========================
def main() -> None:
    project_dir, raw_dir, k_values, window_size_values, s2n_threshold_values = build_runtime_settings()

    validate_parameter_values(k_values, "FRAME_INTERVAL_VALUES")
    validate_parameter_values(window_size_values, "WINDOW_SIZE_PX_VALUES")
    validate_parameter_values(s2n_threshold_values, "SNR_THRESHOLD_VALUES")

    if any(k <= 0 for k in k_values):
        raise ValueError("All FRAME_INTERVAL_VALUES must be positive integers.")
    if any(w <= 0 for w in window_size_values):
        raise ValueError("All WINDOW_SIZE_PX_VALUES must be positive integers.")

    indexed_files = list_images_in_range(raw_dir, START_FRAME, END_FRAME)
    if not indexed_files:
        raise RuntimeError(
            f"No TIFF files found in {raw_dir} for range {START_FRAME}-{END_FRAME}"
        )

    idx_to_path = {idx: path for idx, path in indexed_files}
    image_cache: Dict[Path, np.ndarray] = {}

    total_pairs = 0
    combination_count = 0

    for k in k_values:
        for window_size in window_size_values:
            for s2n_threshold in s2n_threshold_values:
                combination_count += 1
                total_pairs += process_single_combination(
                    project_dir=project_dir,
                    k=k,
                    window_size=window_size,
                    s2n_threshold=s2n_threshold,
                    idx_to_path=idx_to_path,
                    image_cache=image_cache,
                )

    print(
        f"Finished all combinations. "
        f"combinations={combination_count}, total_pairs={total_pairs}"
    )


if __name__ == "__main__":
    main()
