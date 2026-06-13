"""
Task 2: Road Map Generation

Pipeline per video (= one road):
  0. Keep only the best detection per frame (removes headlight-glow ghosts)
  1. Gap-aware outlier removal (allowed jump scales with frame gap)
  2. Arc-length resampling (closes detection gaps, equalizes density)
  3. Savitzky-Golay smoothing (window validated visually)
  4. Endpoint extrapolation so roads start/end near the frame edges
  5. Final resample + light smoothing pass (removes extension joint)

Also synthesizes empty_template.jpg (per-pixel temporal median across all
videos: static camera => moving cars vanish) and uses it as the overlay
background and as the template referenced in the JSON.

Saves roads_defined.json in the format required by the hackathon slides:
{
  "image": {"template_path": "empty_template.jpg", "width": W, "height": H},
  "roads": [{"road_id": 0, "num_trajectories": 1,
             "centerline": [{"x": ..., "y": ...}, ...]}, ...]
}
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ----------------------------- configuration -----------------------------
VIDEO_DIR = Path("./1-individual_vehicles")
CSV_PATH = Path("./results/task1/vehicle_coordinates.csv")
JSON_PATH = Path("./results/task2/roads_defined.json")
TEMPLATE_PATH = Path("./results/task2/empty_template.jpg")
OUT_DIR = Path("./results/task2")

MAX_JUMP_PX = 80          # allowed jump per 1-frame gap
SMOOTH_WINDOW = 41        # Savitzky-Golay window (odd number of points)
SMOOTH_POLY = 3           # Savitzky-Golay polynomial order
RESAMPLE_POINTS = 200     # points per road in the final JSON
EDGE_MARGIN = 5           # how close (px) endpoints should get to the edge
END_DIR_POINTS = 20       # points used to estimate end direction
TOP_LIMIT = 300           # roads stop below the banner; None = true frame top
TEMPLATE_SAMPLES = 25     # frames sampled per video for the empty template
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
# --------------------------------------------------------------------------


def list_videos():
    videos = sorted(p for p in VIDEO_DIR.iterdir()
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos in {VIDEO_DIR.resolve()}")
    return videos


def build_empty_template():
    """Per-pixel temporal median across frames sampled from all videos.
    Static camera => moving vehicles/pedestrians vanish from the median."""
    if TEMPLATE_PATH.exists():
        print(f"Using existing {TEMPLATE_PATH}")
        return cv2.imread(str(TEMPLATE_PATH))

    print("Building empty template (temporal median)...")
    frames = []
    for vp in list_videos():
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(0, max(total - 1, 0),
                               TEMPLATE_SAMPLES, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()

    template = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(TEMPLATE_PATH), template)
    print(f"Saved {TEMPLATE_PATH} from {len(frames)} sampled frames")
    return template


def remove_outliers(xs, ys, frames):
    """Drop points that jump impossibly far from the previous kept point.
    The allowed distance scales with the frame gap, so real points after
    a detection gap are not wrongly discarded."""
    keep_x, keep_y, last_f = [xs[0]], [ys[0]], frames[0]
    for x, y, fidx in zip(xs[1:], ys[1:], frames[1:]):
        gap = max(fidx - last_f, 1)
        d = np.hypot(x - keep_x[-1], y - keep_y[-1])
        if d <= MAX_JUMP_PX * gap:
            keep_x.append(x)
            keep_y.append(y)
            last_f = fidx
    return np.array(keep_x), np.array(keep_y)


def resample_by_arclength(xs, ys, n_points):
    """Resample a polyline to n_points spaced evenly along its length.
    This also closes gaps left by missed detections / removed outliers."""
    d = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] == 0:
        return xs, ys
    s_new = np.linspace(0, s[-1], n_points)
    return np.interp(s_new, s, xs), np.interp(s_new, s, ys)


def extend_to_edge(xs, ys, width, height, top_limit):
    """Linearly extrapolate both ends of the road until they reach near a
    frame edge, using the direction of the last few points."""
    y_top = top_limit if top_limit is not None else EDGE_MARGIN

    def extend(px, py, dx, dy):
        norm = np.hypot(dx, dy)
        if norm < 1e-6:
            return px, py
        dx, dy = dx / norm, dy / norm
        candidates = []
        if dx > 0:
            candidates.append((width - EDGE_MARGIN - px) / dx)
        if dx < 0:
            candidates.append((EDGE_MARGIN - px) / dx)
        if dy > 0:
            candidates.append((height - EDGE_MARGIN - py) / dy)
        if dy < 0:
            candidates.append((y_top - py) / dy)
        pos = [c for c in candidates if c > 0]
        t = min(pos) if pos else 0
        return px + dx * t, py + dy * t

    k = min(END_DIR_POINTS, len(xs) - 1)
    sx, sy = extend(xs[0], ys[0], xs[0] - xs[k], ys[0] - ys[k])
    ex, ey = extend(xs[-1], ys[-1], xs[-1] - xs[-k - 1], ys[-1] - ys[-k - 1])

    xs = np.concatenate([[sx], xs, [ex]])
    ys = np.concatenate([[sy], ys, [ey]])
    return xs, ys


def build_road(group, width, height):
    """Full pipeline for one video -> one smooth road polyline."""
    group = group.sort_values("frame_index")
    xs = group["center_x"].to_numpy(dtype=float)
    ys = group["center_y"].to_numpy(dtype=float)
    frames = group["frame_index"].to_numpy(dtype=int)

    # 1) gap-aware outlier removal
    xs, ys = remove_outliers(xs, ys, frames)

    # 2) even resampling (closes gaps, equalizes point density)
    xs, ys = resample_by_arclength(xs, ys, RESAMPLE_POINTS)

    # 3) smoothing
    window = min(SMOOTH_WINDOW, len(xs) - (1 - len(xs) % 2))  # keep odd
    if window >= 5:
        xs = savgol_filter(xs, window, SMOOTH_POLY)
        ys = savgol_filter(ys, window, SMOOTH_POLY)

    # 4) extend endpoints to the frame edges
    xs, ys = extend_to_edge(xs, ys, width, height, TOP_LIMIT)

    # 5) final resample + light smoothing (removes extension joint)
    xs, ys = resample_by_arclength(xs, ys, RESAMPLE_POINTS)
    xs = savgol_filter(xs, 11, 2)
    ys = savgol_filter(ys, 11, 2)

    # clamp inside the frame
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    return xs, ys


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # 0) one car per individual video -> best detection per frame
    before = len(df)
    df = (df.sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["video_index", "frame_index"])
            .sort_values(["video_index", "frame_index"]))
    print(f"Dedup: removed {before - len(df)} duplicate detections "
          f"({before} -> {len(df)} rows)")

    background = build_empty_template()
    height, width = background.shape[:2]
    print(f"Frame size: {width}x{height}, "
          f"{df.video_index.nunique()} videos in CSV")

    roads_list = []
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0),
              (255, 0, 0), (255, 0, 255), (0, 255, 255)]

    overlay = background.copy()
    plt.figure(figsize=(8, 8))

    for i, (vid, group) in enumerate(sorted(df.groupby("video_index"))):
        xs, ys = build_road(group, width, height)
        roads_list.append({
            "road_id": i,
            "num_trajectories": 1,
            "centerline": [{"x": round(float(x), 3),
                            "y": round(float(y), 3)}
                           for x, y in zip(xs, ys)],
        })
        print(f"  road_id {i}: {len(xs)} pts, "
              f"start=({xs[0]:.0f},{ys[0]:.0f}) end=({xs[-1]:.0f},{ys[-1]:.0f})")

        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(overlay, [pts], False, colors[i % len(colors)], 4)
        cv2.putText(overlay, f"road_{i}", tuple(pts[len(pts) // 2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    colors[i % len(colors)], 3, cv2.LINE_AA)

        plt.scatter(group.center_x, group.center_y, s=2, alpha=0.3)
        plt.plot(xs, ys, linewidth=2, label=f"road_{i}")

    out = {
        "image": {
            "template_path": TEMPLATE_PATH.name,
            "width": width,
            "height": height,
        },
        "roads": roads_list,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(roads_list)} roads to {JSON_PATH.resolve()}")

    cv2.imwrite(str(OUT_DIR / "roadmap_overlay.jpg"), overlay)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Road map: raw detections (dots) vs smoothed roads (lines)")
    plt.savefig(OUT_DIR / "roadmap_plot.png", dpi=150, bbox_inches="tight")
    print(f"Verification images in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()