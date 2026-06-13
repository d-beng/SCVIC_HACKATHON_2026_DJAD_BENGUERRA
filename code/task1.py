"""
Task:
- Auto-measures the black+red banner height (top AND bottom) of every video
  by classifying pixels as banner-like (black or red text), robust to the
  red "Provided by Quanser" text
- Crops banners off each frame BEFORE running YOLOv11
- Rejects phantom oversized boxes (false positives on buildings)
- Shifts coordinates back to original full-frame pixel space
- Saves vehicle_coordinates.csv + annotated frames + annotated videos
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------- configuration -----------------------------
VIDEO_DIR = Path("./1-individual_vehicles")
WEIGHTS_PATH = Path("./yolo_weight.pt")
OUTPUT_DIR = Path("./results/task1/task1_videos")
CSV_PATH = Path("./results/task1/vehicle_coordinates.csv")

CONF_THRESHOLD = 0.25          # keep low: edge/headlight cars can be ~0.4
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# banner measurement
BANNER_SAMPLE_FRAMES = 5       # frames sampled per video
BANNER_ROW_FRACTION = 0.50     # row is banner if >50% pixels are black/red
BANNER_MARGIN = 10             # extra safety pixels added to measured banner

# phantom-box rejection: a real car never covers this much of the frame
MAX_BOX_FRACTION = 0.25

# colors (BGR)
BOX_COLOR = (0, 200, 0)
CENTER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
# --------------------------------------------------------------------------


def find_videos(video_dir: Path):
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir.resolve()}")
    return videos


def measure_banner(video_path: Path):
    """Measure banner heights at the TOP and BOTTOM of a video.

    The banner consists only of black background + red text. A pixel is
    'banner-like' if it is near-black OR strongly red. A row belongs to the
    banner if >BANNER_ROW_FRACTION of its pixels are banner-like in EVERY
    sampled frame (np.minimum across samples), so dark scene content in a
    single frame cannot fool the measurement.

    Returns (top_crop, bottom_crop) in pixels, including a safety margin.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = np.linspace(0, max(total - 1, 0),
                             num=min(BANNER_SAMPLE_FRAMES, max(total, 1)),
                             dtype=int)

    row_frac = None
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        b, g, r = cv2.split(frame.astype(np.int16))
        is_black = (b < 50) & (g < 50) & (r < 50)
        is_red = (r > 90) & (g < 70) & (b < 70)
        banner_pix = (is_black | is_red).astype(np.float32)
        frac = banner_pix.mean(axis=1)   # banner-like fraction per row
        row_frac = frac if row_frac is None else np.minimum(row_frac, frac)
    cap.release()

    if row_frac is None:
        return 0, 0

    h = len(row_frac)

    top = 0
    while top < h and row_frac[top] > BANNER_ROW_FRACTION:
        top += 1

    bottom = 0
    while bottom < h and row_frac[h - 1 - bottom] > BANNER_ROW_FRACTION:
        bottom += 1

    # add safety margin only if a banner was actually found; cap at 1/3 frame
    if top > 0:
        top = min(top + BANNER_MARGIN, h // 3)
    if bottom > 0:
        bottom = min(bottom + BANNER_MARGIN, h // 3)
    return top, bottom


def annotate_frame(frame, detections):
    """Draw bounding boxes, center points, and confidence labels."""
    for (x1, y1, x2, y2, cx, cy, conf) in detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      BOX_COLOR, 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, CENTER_COLOR, -1)
        label = f"vehicle {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1) - th - 8),
                      (int(x1) + tw + 4, int(y1)), BOX_COLOR, -1)
        cv2.putText(frame, label, (int(x1) + 2, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1,
                    cv2.LINE_AA)
    return frame


def process_video(model, video_path: Path, video_index: int, csv_writer):
    top_crop, bottom_crop = measure_banner(video_path)
    print(f"\n=== Video {video_index}: {video_path.name} ===")
    print(f"  banner: top={top_crop}px, bottom={bottom_crop}px (auto-measured)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}, skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = video_path.stem
    frames_dir = OUTPUT_DIR / video_name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = OUTPUT_DIR / video_name / f"{video_name}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_video_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    rejected = 0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # crop banners off before the model sees the frame
        y_end = height - bottom_crop if bottom_crop > 0 else height
        roi = frame[top_crop:y_end, :]

        results = model(roi, conf=CONF_THRESHOLD, verbose=False)

        detections = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # reject phantom boxes far too large to be a car
                w_box, h_box = x2 - x1, y2 - y1
                if (w_box > MAX_BOX_FRACTION * width
                        or h_box > MAX_BOX_FRACTION * height):
                    rejected += 1
                    continue

                # shift coordinates back to full-frame pixel space
                y1 += top_crop
                y2 += top_crop
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append((x1, y1, x2, y2, cx, cy, conf))

                csv_writer.writerow([
                    video_index, video_path.name, frame_index,
                    round(cx, 2), round(cy, 2),
                    round(x1, 2), round(y1, 2),
                    round(x2, 2), round(y2, 2),
                    round(conf, 4),
                ])

        annotated = annotate_frame(frame.copy(), detections)
        cv2.imwrite(str(frames_dir / f"frame_{frame_index:06d}.jpg"),
                    annotated)
        writer.write(annotated)

        if frame_index % 50 == 0:
            print(f"  frame {frame_index}/{total}  "
                  f"({len(detections)} detection(s))")
        frame_index += 1

    cap.release()
    writer.release()
    print(f"  -> rejected {rejected} oversized phantom box(es)")
    print(f"  -> annotated frames: {frames_dir}")
    print(f"  -> annotated video : {out_video_path}")


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH.resolve()}")

    print(f"Loading model {WEIGHTS_PATH} ...")
    model = YOLO(str(WEIGHTS_PATH))

    videos = find_videos(VIDEO_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "video_index", "video_name", "frame_index",
            "center_x", "center_y",
            "x1", "y1", "x2", "y2", "confidence",
        ])
        for video_index, video_path in enumerate(videos):
            process_video(model, video_path, video_index, csv_writer)

    print(f"\nAll done. Coordinates saved to {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()