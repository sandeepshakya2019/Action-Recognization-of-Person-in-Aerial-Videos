import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class KalmanFilter:
    def __init__(self):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.dt = 1.0

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = self.dt

        self.H = np.eye(4, 8)

        self.P = np.eye(8) * 10
        self.Q = np.eye(8)
        self.R = np.eye(4)

        self.x = None

    def initiate(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        z = np.array([cx, cy, w, h])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        ]
    


class DeepFeatureExtractor:
    def extract(self, frame, bbox):
        # Simple placeholder: color histogram
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(128)

        hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return hist
    

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return inter / (a1 + a2 - inter + 1e-6)


def center_distance(b1, b2):
    c1 = [(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2]
    c2 = [(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2]
    return np.linalg.norm(np.array(c1) - np.array(c2))

class Track:
    def __init__(self, bbox, tid, feature):
        self.id = tid
        self.kf = KalmanFilter()
        self.kf.initiate(bbox)

        self.feature = feature

        self.hits = 1
        self.lost = 0
        self.confirmed = False

        self.smooth_bbox = bbox
        self.last_box = bbox   # ✅ FIX

    def predict(self):
        return self.kf.predict()

    def update(self, bbox, feature):
        self.kf.update(bbox)

        alpha = 0.7
        new_bbox = self.kf.get_bbox()

        self.smooth_bbox = [
            alpha * new_bbox[i] + (1 - alpha) * self.smooth_bbox[i]
            for i in range(4)
        ]

        self.last_box = self.smooth_bbox   # ✅ FIX

        self.feature = 0.8 * self.feature + 0.2 * feature

        self.hits += 1
        self.lost = 0

        if self.hits >= 3:
            self.confirmed = True

import numpy as np
from scipy.optimize import linear_sum_assignment

class HybridTracker:
    def __init__(self, max_age=30, match_thresh=0.5):
        self.tracks = {}
        self.next_id = 0

        self.extractor = DeepFeatureExtractor()

        self.max_age = max_age
        self.match_thresh = match_thresh

    def update(self, detections, frame):
        outputs = []

        # -----------------------------
        # Extract features
        # -----------------------------
        det_feats = [self.extractor.extract(frame, d) for d in detections]

        # -----------------------------
        # INIT CASE
        # -----------------------------
        if len(self.tracks) == 0:
            for i, d in enumerate(detections):
                tid = self.next_id
                self.tracks[tid] = Track(d, tid, det_feats[i])

                # FIXED: return confirmed=False
                outputs.append((d, tid, False))

                self.next_id += 1
            return outputs

        track_ids = list(self.tracks.keys())
        preds = [self.tracks[t].predict() for t in track_ids]

        cost = np.ones((len(preds), len(detections)))

        # -----------------------------
        # COST MATRIX
        # -----------------------------
        for i, p in enumerate(preds):
            for j, d in enumerate(detections):

                # Gating (very important)
                if center_distance(p, d) > 200:
                    continue

                iou_score = iou(p, d)
                motion = np.exp(-center_distance(p, d) / 50)
                app = cosine_sim(self.tracks[track_ids[i]].feature, det_feats[j])

                sim = 0.25 * iou_score + 0.25 * motion + 0.5 * app
                cost[i, j] = 1 - sim

        r, c = linear_sum_assignment(cost)

        assigned = set()
        matched_tracks = set()

        # -----------------------------
        # MATCHED TRACKS
        # -----------------------------
        for i, j in zip(r, c):
            if (1 - cost[i, j]) > self.match_thresh:
                tid = track_ids[i]

                self.tracks[tid].update(detections[j], det_feats[j])

                # FIXED: include confirmed flag
                outputs.append((
                    self.tracks[tid].smooth_bbox,
                    tid,
                    self.tracks[tid].confirmed
                ))

                assigned.add(j)
                matched_tracks.add(tid)

        # -----------------------------
        # UNMATCHED TRACKS
        # -----------------------------
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].lost += 1

        # -----------------------------
        # NEW TRACKS
        # -----------------------------
        for j, d in enumerate(detections):
            if j not in assigned:
                tid = self.next_id
                self.tracks[tid] = Track(d, tid, det_feats[j])

                # FIXED: confirmed=False
                outputs.append((d, tid, False))

                self.next_id += 1

        # -----------------------------
        # DELETE DEAD TRACKS
        # -----------------------------
        to_delete = []
        for tid, t in self.tracks.items():
            if t.lost > self.max_age or (not t.confirmed and t.lost > 5):
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return outputs