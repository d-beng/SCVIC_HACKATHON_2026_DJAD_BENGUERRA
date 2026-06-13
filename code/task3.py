"""
Task 3: Consistent tracking, road association, future path estimation

Approach:
  Every tracked vehicle is associated to a road from roads_defined.json and
  a locked direction of travel along it. Future positions are predicted by
  walking the road centerline at the vehicle's speed. That mechanism gives
  the required future-path estimation (dotted points + arrow on the video)
  and the coasting of vehicles lost in collisions (original ID restored).

ID-consistency design:
  - Three-stage matching: (A) active tracks, tight gate, closest first
    (prevents swaps when cars cross); (B) coasting tracks, growing gate;
    (C) RESCUE: leftover detections near a lost track are given to that
    track, never born as a new vehicle.
  - Travel direction along the road is LOCKED while the vehicle is healthy
    and reused while coasting, so a noisy velocity at the moment of loss
    cannot send the prediction walking the wrong way down the road.
  - New IDs: instantly only near a road entry point AND with no lost track
    nearby; anywhere else a candidate must survive MIN_HITS frames.
  - Births, confirmations, rescues and retirements are logged to console
    for debugging.

The 6 roads are drawn semi-transparently (colored + labeled) on every
output frame so vehicle-road association is always visible.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------- configuration -----------------------------
VIDEO_DIR = Path("./2-multiple_vehicles")   # <-- adjust to your folder name
WEIGHTS_PATH = Path("./yolo_weight.pt")
ROADS_PATH = Path("./results/task2/roads_defined.json")
OUTPUT_DIR = Path("./results/task3/task3_videos")
CSV_PATH = Path("./results/task3/tracking_results.csv")

CONF_THRESHOLD = 0.25
MAX_BOX_FRACTION = 0.25
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# banner measurement (same values as working Task 1)
BANNER_SAMPLE_FRAMES = 5
BANNER_ROW_FRACTION = 0.50
BANNER_MARGIN = 10

# tracking parameters
GATE_ACTIVE = 45        # px gate for tracks detected in the previous frame
GATE_GROWTH = 5         # extra px of gate per missed frame while coasting
RESCUE_GATE = 180       # px: unmatched detection near a lost track is given
                        # to that track instead of becoming a new vehicle
MAX_MISSED = 50         # frames a lost track keeps coasting before retiring
MIN_HITS = 6            # frames of life before a mid-scene track gets an ID
ROAD_ENTRY_DIST = 120   # px from a road endpoint = instant-ID spawn zone
DUP_MERGE_DIST = 45     # px: detections closer than this are duplicates
COLLISION_DIST = 55     # px between two track centers => collision flagged
FUTURE_STEPS = 15       # predicted future positions drawn per vehicle
TRAIL_LEN = 40          # drawn trail length (frames)
MAX_SPEED = 25          # px/frame hard clamp (~2x measured p99 of 13px)
ADOPT_GATE = 300        # px: a candidate about to be confirmed first looks
                        # for a lost confirmed vehicle within this range of
                        # its position - if found, it ADOPTS that vehicle's
                        # ID instead of creating a new one ("after
                        # re-detection, restore the vehicle id")
HARD_ROAD_MISSED = 12   # rescue requires the remembered road for the first
                        # N missed frames (collision window: prevents ID
                        # swaps); after that the requirement becomes a soft
                        # penalty, so a car re-detected where two roads run
                        # close together (ambiguous nearest-road) can still
                        # be recaptured instead of being reborn as a new ID
ROAD_MISMATCH_PENALTY = 250   # cost added when a detection's nearest road
                              # differs from the track's road: a lost car is
                              # recaptured on ITS OWN road, preventing two
                              # colliding cars from swapping IDs
ROAD_ALPHA = 0.35       # opacity of the road overlay on output frames

TRACK_COLORS = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 0, 255),
                (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]
ROAD_COLORS = [(0, 100, 0), (130, 0, 75), (139, 0, 0), (0, 140, 255),
               (128, 128, 0), (203, 192, 255)]
# --------------------------------------------------------------------------


# --------------------------- banner (from Task 1) ------------------------
def measure_banner(video_path: Path):
    """Measure black+red banner heights at the TOP and BOTTOM of a video."""
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
        frac = banner_pix.mean(axis=1)
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

    if top > 0:
        top = min(top + BANNER_MARGIN, h // 3)
    if bottom > 0:
        bottom = min(bottom + BANNER_MARGIN, h // 3)
    return top, bottom


# ------------------------------ roads ------------------------------------
class RoadMap:
    """Loads roads_defined.json (slide-compliant schema)."""

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.ids = [int(r["road_id"]) for r in data["roads"]]
        self.polys = [np.array([[p["x"], p["y"]] for p in r["centerline"]],
                               dtype=float) for r in data["roads"]]
        self.endpoints = []
        for p in self.polys:
            self.endpoints.append(p[0])
            self.endpoints.append(p[-1])

    def nearest_road(self, x, y):
        best = (None, 1e9, 0)
        for rid, poly in zip(self.ids, self.polys):
            d = np.hypot(poly[:, 0] - x, poly[:, 1] - y)
            i = int(d.argmin())
            if d[i] < best[1]:
                best = (rid, float(d[i]), i)
        return best

    def direction_at(self, road_id, idx):
        """Unit tangent toward increasing point index."""
        poly = self.polys[self.ids.index(road_id)]
        j = min(idx + 1, len(poly) - 1)
        v = poly[j] - poly[idx]
        n = np.hypot(*v)
        return v / n if n > 1e-6 else np.array([0.0, 0.0])

    def travel_sign(self, road_id, x, y, vx, vy):
        """+1 if the velocity points toward increasing centerline index,
        -1 otherwise, 0 if undecidable (too slow)."""
        if road_id is None or np.hypot(vx, vy) < 0.5:
            return 0
        poly = self.polys[self.ids.index(road_id)]
        d = np.hypot(poly[:, 0] - x, poly[:, 1] - y)
        i = int(d.argmin())
        tang = self.direction_at(road_id, i)
        return 1 if np.dot(tang, [vx, vy]) >= 0 else -1

    def near_entry(self, x, y):
        return any(np.hypot(ex - x, ey - y) < ROAD_ENTRY_DIST
                   for ex, ey in self.endpoints)

    def future_path(self, road_id, x, y, speed, step_sign,
                    n_steps=FUTURE_STEPS):
        """Next n_steps positions (one per future frame): walk the
        centerline from the nearest point at the given speed, in the
        direction given by step_sign (+1/-1, locked by the track)."""
        if road_id is None or step_sign == 0 or speed < 0.5:
            return []
        poly = self.polys[self.ids.index(road_id)]
        d = np.hypot(poly[:, 0] - x, poly[:, 1] - y)
        j = int(d.argmin())

        path, dist_left = [], speed
        while len(path) < n_steps:
            nj = j + step_sign
            if nj < 0 or nj >= len(poly):
                break                       # road ends: vehicle will exit
            seg = np.hypot(*(poly[nj] - poly[j]))
            dist_left -= seg
            if dist_left <= 0:
                path.append(poly[nj])
                dist_left = speed
            j = nj
        return path

    def make_overlay_layer(self, width, height):
        """Pre-render the 6 colored road centerlines + labels once."""
        layer = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for rid, poly in zip(self.ids, self.polys):
            color = ROAD_COLORS[rid % len(ROAD_COLORS)]
            pts = poly.astype(np.int32)
            cv2.polylines(layer, [pts], False, color, 6)
            cv2.polylines(mask, [pts], False, 255, 6)
            mid = pts[len(pts) // 2]
            cv2.putText(layer, f"road_{rid}", (int(mid[0]), int(mid[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            cv2.putText(mask, f"road_{rid}", (int(mid[0]), int(mid[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3, cv2.LINE_AA)
        return layer, mask


def draw_roads(frame, layer, mask):
    """Blend the road layer onto a frame at ROAD_ALPHA, only where drawn."""
    blended = cv2.addWeighted(frame, 1 - ROAD_ALPHA, layer, ROAD_ALPHA, 0)
    out = frame.copy()
    out[mask > 0] = blended[mask > 0]
    return out


# ------------------------------ tracks -----------------------------------
class Track:
    _next_id = 1

    def __init__(self, x, y, box):
        self.id = None
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.box = box
        self.hits = 1
        self.missed = 0
        self.road = None
        self.road_history = []      # ordered list of roads visited
        self.road_dir = 0           # locked travel sign along the road
        self.locked_speed = 0.0     # speed at the moment of loss
        self.trail = [(x, y)]
        self.in_collision = False

    def confirm(self):
        if self.id is None:
            self.id = Track._next_id
            Track._next_id += 1

    def advance(self, roads):
        """Move one frame forward. Recently detected: constant velocity.
        Lost (coasting): walk along the road in the LOCKED direction at
        the speed it had when it was lost."""
        if self.missed > 0 and self.road is not None and self.road_dir != 0:
            fut = roads.future_path(self.road, self.x, self.y,
                                    self.locked_speed, self.road_dir,
                                    n_steps=1)
            if fut:
                self.x, self.y = float(fut[0][0]), float(fut[0][1])
            else:
                self.x += self.vx
                self.y += self.vy
        else:
            self.x += self.vx
            self.y += self.vy
        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LEN:
            self.trail.pop(0)

    def update(self, x, y, box, roads, alpha=0.6):
        """Fuse a matched detection.

        Normal (track was detected last frame): velocity from positions.
        After a loss (rescue / late recapture): the position jump is NOT a
        real velocity - reset velocity along the locked road direction at
        the locked speed instead, so the state cannot explode."""
        recaptured = self.missed > 0
        if recaptured:
            if self.road is not None and self.road_dir != 0:
                d = np.hypot(x - self.x, y - self.y)
                tang = roads.direction_at(
                    self.road,
                    int(np.argmin(np.hypot(
                        roads.polys[roads.ids.index(self.road)][:, 0] - x,
                        roads.polys[roads.ids.index(self.road)][:, 1] - y))))
                self.vx = float(tang[0]) * self.road_dir * self.locked_speed
                self.vy = float(tang[1]) * self.road_dir * self.locked_speed
            # else: keep previous velocity
        elif len(self.trail) >= 2:
            px, py = self.trail[-2]
            self.vx = alpha * (x - px) + (1 - alpha) * self.vx
            self.vy = alpha * (y - py) + (1 - alpha) * self.vy

        # hard clamp: no vehicle moves faster than MAX_SPEED px/frame
        speed = np.hypot(self.vx, self.vy)
        if speed > MAX_SPEED:
            self.vx *= MAX_SPEED / speed
            self.vy *= MAX_SPEED / speed

        self.x, self.y = x, y
        self.box = box
        self.trail[-1] = (x, y)
        self.hits += 1
        self.missed = 0
        # lock direction + speed from the healthy state only
        if not recaptured and not self.in_collision and \
                self.road is not None:
            sign = roads.travel_sign(self.road, x, y, self.vx, self.vy)
            if sign != 0:
                self.road_dir = sign
            self.locked_speed = min(
                max(np.hypot(self.vx, self.vy), 1.0), MAX_SPEED)


# --------------------------- detection step ------------------------------
def detect(model, frame, width, height, top_crop, bottom_crop):
    y_end = height - bottom_crop if bottom_crop > 0 else height
    roi = frame[top_crop:y_end, :]
    results = model(roi, conf=CONF_THRESHOLD, verbose=False)

    dets = []
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_box, h_box = x2 - x1, y2 - y1
            if w_box > MAX_BOX_FRACTION * width or \
               h_box > MAX_BOX_FRACTION * height:
                continue
            y1 += top_crop
            y2 += top_crop
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dets.append((cx, cy, (x1, y1, x2, y2), conf))

    dets.sort(key=lambda d: -d[3])
    kept = []
    for d in dets:
        if all(np.hypot(d[0] - k[0], d[1] - k[1]) > DUP_MERGE_DIST
               for k in kept):
            kept.append(d)
    return kept


# --------------------------- matching stages -----------------------------
def match_stage(tracks, dets, track_ids, det_ids, gate_fn, roads,
                frame_index, tag, det_roads=None):
    """Greedy lowest-cost-first matching restricted to the given subsets.

    Cost = euclidean distance + ROAD_MISMATCH_PENALTY if the detection's
    nearest road differs from the track's associated road. Gating is on
    raw distance; the penalty only reorders preferences, so a lost car is
    recaptured on its own road first - this is what prevents two colliding
    cars from swapping IDs at re-emergence."""
    pairs = []
    for ti in track_ids:
        t = tracks[ti]
        gate = gate_fn(t)
        for di in det_ids:
            cx, cy = dets[di][0], dets[di][1]
            d = np.hypot(cx - t.x, cy - t.y)
            if d < gate:
                cost = d
                if det_roads is not None and t.road is not None and \
                        det_roads[di] != t.road:
                    if tag == "RESCUE" and t.missed <= HARD_ROAD_MISSED:
                        continue   # HARD inside the collision window
                    cost += ROAD_MISMATCH_PENALTY
                pairs.append((cost, d, ti, di))
    pairs.sort()
    pairs = [(d, ti, di) for cost, d, ti, di in pairs]
    used_t, used_d = set(), set()
    for d, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        cx, cy, box, conf = dets[di]
        t = tracks[ti]
        if tag == "RESCUE" and t.id is not None:
            print(f"    [f{frame_index}] RESCUED id{t.id} "
                  f"after {t.missed} missed (dist {d:.0f}px)")
        t.update(cx, cy, box, roads)
    return used_t, used_d


# ----------------------------- main loop ---------------------------------
def process_video(model, roads, video_path, video_index, csv_writer):
    top_crop, bottom_crop = measure_banner(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    road_layer, road_mask = roads.make_overlay_layer(width, height)

    name = video_path.stem
    frames_dir = OUTPUT_DIR / name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_DIR / name / f"{name}_tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    Track._next_id = 1
    tracks = []
    collision_events = []
    print(f"\n=== Video {video_index}: {video_path.name} ({total} frames) ===")
    print(f"  banner: top={top_crop}px, bottom={bottom_crop}px")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) advance every track one frame
        for t in tracks:
            t.advance(roads)

        # 2) detect
        dets = detect(model, frame, width, height, top_crop, bottom_crop)
        all_d = set(range(len(dets)))
        det_roads = [roads.nearest_road(d[0], d[1])[0] for d in dets]

        # 3) Matching. CONFIRMED tracks always outrank unconfirmed
        #    candidates, so a candidate (often collision debris) can never
        #    steal a detection from a lost real vehicle.
        # Stage A: confirmed active tracks, tight gate (prevents swaps)
        active = [i for i, t in enumerate(tracks)
                  if t.id is not None and t.missed == 0]
        ua, da = match_stage(tracks, dets, active, all_d,
                             lambda t: GATE_ACTIVE, roads,
                             frame_index, "ACTIVE")

        # Stage B: confirmed coasting tracks, growing gate
        coasting = [i for i, t in enumerate(tracks)
                    if t.id is not None and t.missed > 0 and i not in ua]
        ub, db = match_stage(tracks, dets, coasting, all_d - da,
                             lambda t: GATE_ACTIVE + GATE_GROWTH * t.missed,
                             roads, frame_index, "COAST", det_roads)

        # Stage C: RESCUE - leftover detections near a lost confirmed track
        #    belong to that track, never to a new car
        still_lost = [i for i, t in enumerate(tracks)
                      if t.id is not None and t.missed > 0 and i not in ub]
        uc, dc = match_stage(tracks, dets, still_lost, all_d - da - db,
                             lambda t: RESCUE_GATE, roads,
                             frame_index, "RESCUE", det_roads)

        # Stage D: unconfirmed candidates get only what is left
        cand = [i for i, t in enumerate(tracks) if t.id is None]
        ud, dd = match_stage(tracks, dets, cand, all_d - da - db - dc,
                             lambda t: GATE_ACTIVE, roads,
                             frame_index, "CAND")

        used_t = ua | ub | uc | ud
        used_d = da | db | dc | dd

        # 6) unmatched tracks: count the miss; retire expired ones (logged)
        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.missed += 1
        survivors = []
        for t in tracks:
            off_frame = not (0 <= t.x < width and 0 <= t.y < height)
            if t.missed <= MAX_MISSED and not off_frame:
                survivors.append(t)
            elif t.id is not None:
                print(f"    [f{frame_index}] RETIRED id{t.id} "
                      f"at ({t.x:.0f},{t.y:.0f}) road_{t.road} "
                      f"after {t.missed} missed")
        tracks = survivors

        # 7) unmatched detections: new candidate tracks (no ID yet)
        for di in sorted(all_d - used_d):
            cx, cy, box, conf = dets[di]
            tracks.append(Track(cx, cy, box))

        # 8) road association + ID confirmation. Instant ID only at a road
        #    entry with NO lost track nearby; otherwise survive MIN_HITS.
        #    ROAD MEMORY: a track's road is only (re)assigned while the
        #    vehicle is healthy - detected this frame and not inside a
        #    collision. While lost or colliding, the road is FROZEN, so the
        #    vehicle 'remembers' which road it was on and can only be
        #    recaptured there. This is what makes post-collision ID swaps
        #    (nearly) impossible.
        for t in tracks:
            healthy = (t.missed == 0 and not t.in_collision)
            if t.road is None or healthy:
                rid, rdist, ridx = roads.nearest_road(t.x, t.y)
                t.road = rid
                if not t.road_history or t.road_history[-1] != rid:
                    t.road_history.append(rid)
            if t.id is None:
                # entry fast-track only if NO confirmed vehicle (lost or
                # tracked) is anywhere near: a detection appearing next to
                # an existing vehicle in an entry/exit zone is a duplicate
                # (headlight beam, partial box), never a new car. Genuine
                # simultaneous entries simply wait MIN_HITS frames.
                confirmed_nearby = any(
                    o.id is not None and
                    np.hypot(o.x - t.x, o.y - t.y) < ADOPT_GATE
                    for o in tracks)
                if t.hits >= MIN_HITS or (roads.near_entry(t.x, t.y)
                                          and not confirmed_nearby):
                    # ADOPTION: before minting a new ID, look for a lost
                    # confirmed vehicle nearby. A track earning an ID while
                    # another vehicle is lost in the vicinity IS that
                    # vehicle re-detected - restore its ID, do not create.
                    donor, best = None, ADOPT_GATE
                    for o in tracks:
                        if o is t or o.id is None or o.missed == 0:
                            continue
                        d = np.hypot(o.x - t.x, o.y - t.y)
                        if d < best:
                            donor, best = o, d
                    if donor is not None:
                        t.id = donor.id
                        t.road = donor.road
                        t.road_dir = donor.road_dir
                        t.locked_speed = donor.locked_speed
                        donor.id = None          # retired silently below
                        donor.missed = MAX_MISSED + 1
                        print(f"    [f{frame_index}] ADOPTED id{t.id} "
                              f"at ({t.x:.0f},{t.y:.0f}) road_{t.road} "
                              f"(dist {best:.0f}px) - id restored")
                    else:
                        t.confirm()
                        why = ("entry" if roads.near_entry(t.x, t.y)
                               else f"{t.hits} hits")
                        print(f"    [f{frame_index}] NEW id{t.id} "
                              f"at ({t.x:.0f},{t.y:.0f}) road_{t.road} "
                              f"({why})")

        # 9) collision detection (bonus)
        confirmed = [t for t in tracks if t.id is not None]
        frame_collisions = set()
        for i in range(len(confirmed)):
            for j in range(i + 1, len(confirmed)):
                a, b = confirmed[i], confirmed[j]
                if np.hypot(a.x - b.x, a.y - b.y) < COLLISION_DIST:
                    frame_collisions.update([a.id, b.id])
                    collision_events.append((frame_index, a.id, b.id))
        for t in confirmed:
            t.in_collision = t.id in frame_collisions

        # 10) write CSV rows + draw
        annotated = draw_roads(frame, road_layer, road_mask)
        for t in confirmed:
            state = "detected" if t.missed == 0 else "predicted"
            csv_writer.writerow([
                video_index, video_path.name, frame_index, t.id, t.road,
                round(t.x, 2), round(t.y, 2), state, int(t.in_collision),
            ])

            color = TRACK_COLORS[(t.id - 1) % len(TRACK_COLORS)]
            x1, y1, x2, y2 = t.box
            bx1, by1 = int(x1), int(y1)
            bx2, by2 = int(x2), int(y2)

            # bounding box: dark thin frame (red when colliding)
            box_color = (0, 0, 255) if t.in_collision else (60, 40, 70)
            if t.missed == 0:
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                              box_color, 2)
            else:
                cv2.circle(annotated, (int(t.x), int(t.y)), 14, color, 2)

            # label bar: dark filled rect, white text 'ID n HR[..] Pk'
            hr = ",".join(str(r) for r in t.road_history[-2:])
            label = f"ID {t.id} HR[{hr}] P{t.road}" + \
                    (" PRED" if t.missed else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.65, 2)
            ly = max(by1 - 10, th + 12)
            cv2.rectangle(annotated, (bx1 - 2, ly - th - 8),
                          (bx1 + tw + 8, ly + 4), (40, 25, 50), -1)
            cv2.putText(annotated, label, (bx1 + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # H marker: small white circle at the vehicle center
            cv2.circle(annotated, (int(t.x), int(t.y)), 6,
                       (255, 255, 255), 2)
            cv2.putText(annotated, f"H{t.id}",
                        (int(t.x) + 9, int(t.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # history trail: solid colored line
            pts = np.array(t.trail, dtype=np.int32)
            cv2.polylines(annotated, [pts], False, color, 3)

            # future path: DOTTED colored line + 'F-Rk' label at its end
            speed = max(np.hypot(t.vx, t.vy), t.locked_speed)
            sign = t.road_dir if t.road_dir != 0 else \
                roads.travel_sign(t.road, t.x, t.y, t.vx, t.vy)
            fut = roads.future_path(t.road, t.x, t.y, speed, sign,
                                    n_steps=FUTURE_STEPS)
            for p in fut:
                cv2.circle(annotated, (int(p[0]), int(p[1])), 4, color, -1)
            if fut:
                fx, fy = int(fut[-1][0]), int(fut[-1][1])
                cv2.putText(annotated, f"F-R{t.road}", (fx + 8, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            cv2.LINE_AA)

        # collision banner: unmissable strip when a collision is active
        if frame_collisions:
            ids_txt = " & ".join(f"ID {i}"
                                 for i in sorted(frame_collisions))
            btxt = f"!! COLLISION DETECTED: {ids_txt} !!"
            (tw, th), _ = cv2.getTextSize(btxt, cv2.FONT_HERSHEY_SIMPLEX,
                                          1.4, 4)
            by = top_crop + 20
            cv2.rectangle(annotated, (0, by),
                          (width, by + th + 28), (0, 0, 180), -1)
            cv2.putText(annotated, btxt,
                        ((width - tw) // 2, by + th + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (255, 255, 255), 4, cv2.LINE_AA)

        cv2.imwrite(str(frames_dir / f"frame_{frame_index:06d}.jpg"),
                    annotated)
        writer.write(annotated)

        if frame_index % 50 == 0:
            print(f"  frame {frame_index}/{total}: "
                  f"{len(confirmed)} tracked, {len(dets)} detected")
        frame_index += 1

    cap.release()
    writer.release()
    if collision_events:
        first = {}
        for fidx, a, b in collision_events:
            key = tuple(sorted((a, b)))
            first.setdefault(key, fidx)
        for (a, b), fidx in first.items():
            print(f"  COLLISION: id{a} x id{b} first at frame {fidx}")
    else:
        print("  no collisions detected")
    print(f"  -> outputs in {OUTPUT_DIR / name}")


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH.resolve()}")
    if not ROADS_PATH.exists():
        raise FileNotFoundError(f"Road map not found: {ROADS_PATH.resolve()} "
                                "- run Task 2 first")

    model = YOLO(str(WEIGHTS_PATH))
    roads = RoadMap(ROADS_PATH)
    print(f"Loaded {len(roads.ids)} roads from {ROADS_PATH}")

    videos = sorted(p for p in VIDEO_DIR.iterdir()
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos in {VIDEO_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_index", "video_name", "frame_index",
                    "vehicle_id", "associated_road", "x", "y",
                    "state", "collision"])
        for vi, vp in enumerate(videos):
            process_video(model, roads, vp, vi, w)

    print(f"\nAll done. Results in {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()