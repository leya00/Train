import os
import cv2
import torch
import numpy as np
import warnings
from collections import deque
from scipy.spatial import distance


warnings.filterwarnings("ignore")

# 1) COCO class names
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'bunny', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# random colors for drawing
COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

# 2) Metrics tracker
class DetectionMatrix:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.class_matrix = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(CLASS_NAMES))}

    def update(self, pred_boxes, true_boxes, iou_threshold=0.5):
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(true_boxes, torch.Tensor):
            true_boxes = true_boxes.cpu().numpy()

        matched_preds = np.zeros(len(pred_boxes), bool)
        matched_gts = np.zeros(len(true_boxes), bool)

        # IoU matrix
        iou = np.zeros((len(pred_boxes), len(true_boxes)))
        for i, p in enumerate(pred_boxes):
            for j, g in enumerate(true_boxes):
                iou[i, j] = self._iou(p[:4], g[:4])

        # match by IoU + class
        for i in range(len(pred_boxes)):
            for j in range(len(true_boxes)):
                if not matched_gts[j] and iou[i, j] >= iou_threshold:
                    if int(pred_boxes[i, 5]) == int(true_boxes[j, 5]):
                        cls = int(pred_boxes[i, 5])
                        self.true_positives += 1
                        self.class_matrix[cls]['tp'] += 1
                        matched_preds[i] = True
                        matched_gts[j] = True
                        break

        # remaining unmatched preds = false positives
        for i in range(len(pred_boxes)):
            if not matched_preds[i]:
                cls = int(pred_boxes[i, 5])
                self.false_positives += 1
                self.class_matrix[cls]['fp'] += 1

        # remaining unmatched gts = false negatives
        for j in range(len(true_boxes)):
            if not matched_gts[j]:
                cls = int(true_boxes[j, 5])
                self.false_negatives += 1
                self.class_matrix[cls]['fn'] += 1

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def get_f1_score(self):
        p = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        r = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        return 2 * (p * r) / (p + r)

    def get_class_matrices(self):
        out = {}
        for idx, cnt in self.class_matrix.items():
            tp = cnt['tp']
            fp = cnt['fp']
            fn = cnt['fn']
            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            f = 2 * (p * r) / (p + r + 1e-6)
            out[CLASS_NAMES[idx]] = {'precision': p, 'recall': r, 'f1': f, 'support': tp + fn}
        return out

# 3) Smoother for bounding boxes (optional)
class SmoothBoxTracker:
    def __init__(self, max_history=5, lock_threshold=0.8, stability_threshold=5):
        self.box_history = deque(maxlen=max_history)
        self.lock_threshold = lock_threshold
        self.stability_threshold = stability_threshold
        self.stable_frames = 0
        self.locked = False
        self.last_box = None
        self.fade = 0
        self.max_fade = 10

    def update(self, box):
        if box is not None:
            self.box_history.append(box)
            mean_box = np.mean(self.box_history, axis=0)
            if box[4] >= self.lock_threshold:
                self.stable_frames += 1
                if self.stable_frames >= self.stability_threshold:
                    self.locked = True
                    self.last_box = mean_box.copy()
            else:
                self.stable_frames = max(0, self.stable_frames - 1)
            self.fade = min(self.fade + 1, self.max_fade)
        else:
            self.fade = max(self.fade - 1, 0)

        if self.locked and self.last_box is not None:
            return self.last_box if self.fade > 0 else None
        return mean_box if self.fade > 0 else None

    def get_alpha(self):
        return min(1.0, self.fade / self.max_fade + (0.3 if self.locked else 0))

# 4) Main detection function
def detect_objects_video(
    video_path: str,
    output_path: str = None,
    confidence_threshold: float = 0.5,
    target_classes: list[str] = None,
    gt_annotations: dict[int, list] = None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    model.eval()

    target_idx = [CLASS_NAMES.index(c.lower()) for c in target_classes] if target_classes else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_path:
        dirpath = os.path.dirname(output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    metrics = DetectionMatrix()
    frames_with_trains = 0
    total_train_detections = 0
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = model(rgb)

        preds = results.xyxy[0]
        pred_np = preds.cpu().numpy()

        train_cls = CLASS_NAMES.index('train')
        if (pred_np[:, 5] == train_cls).any():
          frames_with_trains += 1
          total_train_detections += int((pred_np[:, 5] == train_cls).sum())

        if gt_annotations:
            metrics.update(preds, gt_annotations.get(frame_i, []))

        if output_path:
            out.write(frame)

        frame_i += 1

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    # final metrics
    stats = {
      'frames_with_trains': frames_with_trains,
      'total_train_detections': total_train_detections,
      'f1_score': metrics.get_f1_score(),
      'precision': metrics.true_positives / (metrics.true_positives + metrics.false_positives + 1e-6),
      'recall': metrics.true_positives / (metrics.true_positives + metrics.false_negatives + 1e-6),}
    
    class_metrics = metrics.get_class_matrices()

    return {
        'statistics': stats,
        'per_class': class_metrics
    }
