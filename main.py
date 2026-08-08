import cv2
import numpy as np
import math
from ultralytics import YOLO
from sklearn.cluster import KMeans

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "yolov8n.pt"       # YOLOv8 model
VIDEO_PATH = "video1.mp4"       # input video
CONFIDENCE_THRESHOLD = 0.5
ANGLE_MAX_DEG = 30.0
CENTER_MAX_RATIO = 0.5
WEIGHT_CENTER = 0.45
WEIGHT_ANGLE = 0.45
WEIGHT_INSIDE = 0.10
SHOW_DEBUG = True

# Average car size (pixels)
CAR_W = 180
CAR_H = 360
MAX_SLOTS = 10  # maximum clusters/slots to detect

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_model(path):
    return YOLO(path)

def get_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def safe_min_area_angle(box_pts):
    pts = np.array(box_pts, dtype=np.float32)
    if pts.shape[0] < 3:
        return 0.0
    rect = cv2.minAreaRect(pts)
    angle = rect[2]
    w, h = rect[1]
    if w < h:
        angle += 90
    return angle % 180

def minimal_angle_diff_deg(a, b):
    diff = abs(a - b) % 180
    return 180 - diff if diff > 90 else diff

def normalize(val, max_val):
    if max_val <= 0:
        return 1.0 if val <= 0 else 0.0
    return max(0.0, min(1.0, val / max_val))

def create_rotated_slot(cx, cy, width, height, angle_deg):
    """Return rotated rectangle polygon points"""
    angle_rad = np.deg2rad(angle_deg)
    dx = width / 2
    dy = height / 2
    corners = np.array([[-dx, -dy],[dx, -dy],[dx, dy],[-dx, dy]])
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    rotated = np.dot(corners, R.T)
    rotated[:,0] += cx
    rotated[:,1] += cy
    return rotated.astype(np.float32)

def perpendicular_distance(point, line_p1, line_p2):
    """Distance from point to line defined by two points"""
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den != 0 else float('inf')

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# VIDEO CAPTURE
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

slot_meta = {}
next_slot_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    cars = []

    # Collect detected cars (class 2)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            try:
                cls = int(box.cls[0])
            except:
                continue
            if cls == 2:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = get_center(x1, y1, x2, y2)
                cars.append({
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "box_pts": np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                })

    # ----------------------------
    # AUTO SLOT DETECTION (first frame or empty)
    # ----------------------------
    if len(cars) > 0 and len(slot_meta) == 0:
        car_centers = np.array([c['center'] for c in cars])
        n_clusters = min(MAX_SLOTS, len(cars))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(car_centers)

        for lbl in range(n_clusters):
            cluster_points = car_centers[labels == lbl]
            if len(cluster_points) == 0:
                continue
            cx, cy = cluster_points.mean(axis=0)
            # average car angles in cluster
            car_angles = [safe_min_area_angle(cars[i]['box_pts']) for i in range(len(cars)) if labels[i]==lbl]
            avg_angle = np.mean(car_angles) if car_angles else 0.0
            poly = create_rotated_slot(cx, cy, CAR_W, CAR_H, avg_angle)
            slot_meta[next_slot_id] = {
                "poly": poly,
                "centroid": (cx, cy),
                "orientation": avg_angle,
                "half_width": CAR_W/2
            }
            next_slot_id += 1

    # ----------------------------
    # ASSIGN CARS TO SLOTS
    # ----------------------------
    slot_assignments = {sid: [] for sid in slot_meta.keys()}
    for c in cars:
        cx, cy = c['center']
        best_sid = None
        best_dist = float('inf')
        for sid, meta in slot_meta.items():
            sx, sy = meta['centroid']
            d = math.hypot(cx - sx, cy - sy)
            if d < best_dist:
                best_dist = d
                best_sid = sid
        slot_assignments[best_sid].append(c)

    # ----------------------------
    # EVALUATE & DRAW SLOTS
    # ----------------------------
    for sid, meta in slot_meta.items():
        poly = meta["poly"]
        centroid = meta["centroid"]
        slot_half_width = meta["half_width"]
        assigned = slot_assignments.get(sid, [])
        cx_s, cy_s = int(centroid[0]), int(centroid[1])

        cv2.polylines(frame, [poly.astype(np.int32)], True, (255,255,0), 2)

        if len(assigned) == 0:
            cv2.putText(frame, f"Slot {sid}: NO CAR", (cx_s-40, cy_s), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            continue

        if len(assigned) > 1:
            assigned = sorted(assigned, key=lambda c: math.hypot(c["center"][0]-centroid[0], c["center"][1]-centroid[1]))
        car = assigned[0]
        x1,y1,x2,y2 = car["bbox"]
        car_cx, car_cy = car["center"]

        car_angle = safe_min_area_angle(car["box_pts"])
        angle_diff = minimal_angle_diff_deg(car_angle, meta["orientation"])
        angle_norm = normalize(angle_diff, ANGLE_MAX_DEG)

        # compute perpendicular distance to slot orientation line
        poly_pts = poly.astype(np.float32)
        line_p1, line_p2 = poly_pts[0], poly_pts[1]  # top edge as reference
        center_dev_px = perpendicular_distance((car_cx, car_cy), line_p1, line_p2)
        center_dev_norm = normalize(center_dev_px, slot_half_width * CENTER_MAX_RATIO)

        inside = cv2.pointPolygonTest(poly.astype(np.int32), (car_cx, car_cy), False)
        inside_score = 1.0 if inside >= 0 else 0.0

        center_conf = max(0.0, 1.0 - center_dev_norm)
        angle_conf = max(0.0, 1.0 - angle_norm)
        confidence = WEIGHT_CENTER*center_conf + WEIGHT_ANGLE*angle_conf + WEIGHT_INSIDE*inside_score
        aligned = confidence >= CONFIDENCE_THRESHOLD

        color = (0,255,0) if aligned else (0,0,255)
        label = f"Slot {sid}: {'ALIGNED' if aligned else 'MISALIGNED'} ({confidence:.2f})"

        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.circle(frame, (int(car_cx), int(car_cy)), 4, color, -1)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if SHOW_DEBUG:
            dbg1 = f"angCar={car_angle:.1f} dAng={angle_diff:.1f}"
            dbg2 = f"ctrDevPx={center_dev_px:.1f} norm={center_dev_norm:.2f} inside={int(inside)}"
            cv2.putText(frame, dbg1, (int(x1), int(y2)+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, dbg2, (int(x1), int(y2)+36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    cv2.imshow("SmartPark - Auto Rotated Slots", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(100)
cv2.destroyAllWindows()
