# detector_onnx.py
import os, cv2, numpy as np, onnxruntime as ort

COCO80 = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
 "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
 "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
 "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
 "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
 "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
 "hair drier","toothbrush"
]

# === Umschalten für deinen Test ===
TARGET_CLASS = "sports ball"       # später zurück auf "sports ball"
DEBUG_SHOW_ANY = True         # zeigt notfalls die beste beliebige Klasse

def _letterbox(img, new_size=640, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top  = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

class YoloOnnxDetector:
    def __init__(self, model_path: str, imgsz: int = 640, conf_thres: float = 0.35, iou_thres: float = 0.50):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model missing: {model_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(2, (os.cpu_count() or 4) // 2)

        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.inp_name = inp.name

        # Modell-Inputgröße auslesen (fix vs. dynamisch)
        shape = list(inp.shape)
        try:
            h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else None
            w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else None
        except Exception:
            h = w = None
        if h and w and h == w:
            self.imgsz = h
            print(f"[i] Modell erwartet feste Größe: {h}x{w} – imgsz darauf gesetzt.")
        else:
            self.imgsz = int(imgsz)
            print(f"[i] Modell erlaubt dynamische Größe – imgsz={self.imgsz}.")

        self.conf_thres = float(conf_thres)
        self.iou_thres  = float(iou_thres)

        # Warm-up (beschleunigt erste echte Inferenz)
        dummy = np.zeros((1,3,self.imgsz,self.imgsz), np.float32)
        _ = self.session.run(None, {self.inp_name: dummy})

        # Zielklasse vorbereiten
        self.target_idx = COCO80.index(TARGET_CLASS)

    def _nms_first(self, boxes_xywh, scores):
        # boxes: [x1,y1,w,h] (letterbox coords, float), scores: (N,)
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.astype(np.float32).tolist(),
            scores=scores.astype(float).tolist(),
            score_threshold=float(self.conf_thres),
            nms_threshold=float(self.iou_thres)
        )
        if len(idxs) == 0:
            return None
        if isinstance(idxs, (list, tuple)):
            return int(idxs[0])
        idxs = np.array(idxs).reshape(-1)
        return int(idxs[0]) if idxs.size > 0 else None

    def detect_ball(self, frame_bgr):
        # --- Preprocess ---
        img_lb, scale, pad_x, pad_y = _letterbox(frame_bgr, self.imgsz)
        img = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]  # NCHW
        img = np.ascontiguousarray(img, dtype=np.float32)

        # --- Inference ---
        pred = self.session.run(None, {self.inp_name: img})[0]  # (1,84/85,8400) oder (1,8400,84/85)
        if pred.ndim != 3:
            return None
        if pred.shape[1] in (84, 85):
            pred = np.transpose(pred, (0,2,1))
        pred = pred[0]  # (N, 84/85)

        last_dim = pred.shape[1]
        if last_dim == 85:
            boxes_xywh = pred[:, :4]
            obj_conf   = pred[:, 4:5]                  # (N,1)
            cls_probs  = pred[:, 5:]                   # (N,80)
        elif last_dim == 84:
            boxes_xywh = pred[:, :4]
            obj_conf   = np.ones((pred.shape[0],1), dtype=pred.dtype)
            cls_probs  = pred[:, 4:]
        else:
            return None

        # --- Skala prüfen: sind Boxen wahrscheinlich normalisiert? ---
        # Heuristik: Wenn median(w) sehr klein (< 2), dann *wahrscheinlich* 0..1 skaliert.
        if np.median(boxes_xywh[:,2]) <= 2.0:
            boxes_xywh = boxes_xywh * float(self.imgsz)

        # Zielklasse-Scores
        scores_target = (obj_conf * cls_probs[:, self.target_idx:self.target_idx+1]).squeeze(1)
        mask = scores_target >= self.conf_thres

        # Kandidaten-Boxen (letterbox coords, xywh)
        x, y, w, h = boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,2], boxes_xywh[:,3]
        x1 = x - w/2; y1 = y - h/2

        # --- Pfad 1: Zielklasse vorhanden ---
        if np.any(mask):
            sel_w  = w[mask]; sel_h = h[mask]
            sel_x1 = x1[mask]; sel_y1 = y1[mask]
            sel_sc = scores_target[mask]

            nms_boxes = np.stack([sel_x1, sel_y1, sel_w, sel_h], axis=1)
            pick = self._nms_first(nms_boxes, sel_sc)
            if pick is None:
                return None

            bx1, by1, bw, bh = nms_boxes[pick]
            bx2, by2 = bx1 + bw, by1 + bh
            sc = float(sel_sc[pick])
            cls_name = COCO80[self.target_idx]

        # --- Pfad 2: Debug-Fallback – beste beliebige Klasse zeigen ---
        elif DEBUG_SHOW_ANY:
            all_scores = (obj_conf * cls_probs).max(axis=1)          # (N,)
            top_idx = int(all_scores.argmax())
            bx1 = float(x1[top_idx]); by1 = float(y1[top_idx])
            bw  = float(w[top_idx]);  bh  = float(h[top_idx])
            bx2, by2 = bx1 + bw, by1 + bh
            sc = float(all_scores[top_idx])
            cls_name = COCO80[int(cls_probs[top_idx].argmax())]
        else:
            return None

        # --- Zurück in Original-Koordinaten ---
        bx1 -= pad_x; bx2 -= pad_x; by1 -= pad_y; by2 -= pad_y
        bx1 /= scale; bx2 /= scale; by1 /= scale; by2 /= scale
        H, W = frame_bgr.shape[:2]
        bx1 = max(0, min(W-1, bx1)); bx2 = max(0, min(W-1, bx2))
        by1 = max(0, min(H-1, by1)); by2 = max(0, min(H-1, by2))

        cx = int((bx1 + bx2) / 2); cy = int((by1 + by2) / 2)
        r  = int(0.5 * max(bx2 - bx1, by2 - by1))
        return {"x": cx, "y": cy, "r": r, "score": sc,
                "box": (int(bx1), int(by1), int(bx2), int(by2)),
                "cls": cls_name}
