# run.py
import os, sys, time, cv2, yaml, requests
from detector_onnx import YoloOnnxDetector


def _escape_tag(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ").replace("=", "\\=")


class InfluxLineClient:
    def __init__(self, url: str, org: str, bucket: str, token: str):
        base = url.rstrip("/")
        self.write_url = f"{base}/api/v2/write"
        self.params = {"org": org, "bucket": bucket, "precision": "ns"}
        self.headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "text/plain; charset=utf-8"
        }

    def write_detection(self, camera_id: int, detection: dict):
        label = detection.get("cls", "")
        line = (
            f"detections,camera_id={camera_id},label={_escape_tag(label)} "
            f"x={int(detection.get('x', 0))}i,y={int(detection.get('y', 0))}i,"
            f"r={int(detection.get('r', 0))}i,score={float(detection.get('score', 0.0))} "
            f"{time.time_ns()}"
        )
        try:
            resp = requests.post(self.write_url, params=self.params, data=line, headers=self.headers, timeout=2)
            if not resp.ok:
                print(f"[WARN] Influx write failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"[WARN] Influx write error: {e}")

def load_cfg(path="config.yaml"):
    if not os.path.exists(path):
        print("[WARN] config.yaml nicht gefunden – nutze Defaultwerte.")
        return {
            "camera_id": 1,
            "imgsz": 320,
            "conf": 0.25,
            "iou": 0.5,
            "mqtt": {"host": "127.0.0.1", "port": 1883, "topic": "tracker/ball"},
            "log_level": "INFO",
            "influx": {
                "enabled": False,
                "url": "http://localhost:8086",
                "org": "example-org",
                "bucket": "detections",
                "token": ""
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg     = load_cfg()
    cam_id  = int(cfg.get("camera_id", 1))
    imgsz   = int(cfg.get("imgsz", 320))      # <= kleiner = schneller (z.B. 320)
    conf_th = float(cfg.get("conf", 0.25))    # etwas sensibler
    iou_th  = float(cfg.get("iou", 0.5))
    influx_cfg = cfg.get("influx", {}) or {}
    influx_client = None
    if influx_cfg.get("enabled"):
        influx_client = InfluxLineClient(
            url=influx_cfg.get("url", "http://localhost:8086"),
            org=influx_cfg.get("org", "example-org"),
            bucket=influx_cfg.get("bucket", "detections"),
            token=influx_cfg.get("token", "")
        )
        print(f"[i] InfluxDB aktiviert: {influx_cfg.get('url')} | bucket={influx_cfg.get('bucket')} | org={influx_cfg.get('org')}")

    model_path = os.path.join("models", "yolov8n.onnx")
    print(f"[i] Lade Modell: {model_path} | imgsz={imgsz} conf={conf_th} iou={iou_th}")
    det = YoloOnnxDetector(model_path, imgsz=imgsz, conf_thres=conf_th, iou_thres=iou_th)

    # Kamera öffnen – MSMF ist bei dir stabil. MJPG spart USB-Bandbreite.
    print(f"[i] Öffne Kamera {cam_id} (MSMF) ...")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        sys.exit(f"[ERR] Kamera {cam_id} nicht verfügbar.")

    # Auflösung kompakt für CPU, MJPG erzwingen
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # (optional) kleine Buffergröße, falls Treiber unterstützt:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Inference-Skipping: nur jedes N-te Frame inferieren
    INFER_EVERY_DEFAULT = 2   # 2 = jedes zweite Frame; bei Bedarf 3 testen
    skip_enabled        = True
    infer_every         = INFER_EVERY_DEFAULT
    frame_idx           = 0
    last_res            = None

    fps = 0.0
    fps_counter = 0
    fps_t0 = time.time()
    print("[i] Start – ESC schließt Fenster.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame capture fehlgeschlagen.")
            break

        frame_idx += 1
        new_detection = False

        # Nur jedes N-te Frame inferieren (Tracking-by-Detection)
        if (frame_idx % infer_every) == 0:
            last_res = det.detect_ball(frame)
            new_detection = True
        res = last_res

        # Overlay
        if res:
            x, y, r = res["x"], res["y"], res["r"]
            score   = res["score"]
            label   = res.get("cls", "")
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if influx_client and new_detection:
                influx_client.write_detection(cam_id, res)


        # FPS glätten (Exponentielles Mittel)
        # FPS über echte Zeitbasis zählen (robust trotz Frame-Skipping)
        fps_counter += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps = fps_counter / (now - fps_t0)
            fps_counter = 0
            fps_t0 = now

        # HUD
        skip_txt = f"{infer_every} (on)" if skip_enabled else "off"
        hud = f"FPS:{fps:.1f} | imgsz:{imgsz} | skip:{skip_txt} | 640x480 MJPG"
        cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AI Ball Tracker (ESC beendet)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key in (ord('s'), ord('S')):
            skip_enabled = not skip_enabled
            infer_every = INFER_EVERY_DEFAULT if skip_enabled else 1
            frame_idx = 0  # sauber neu zählen für modulo
            state = "aktiviert" if skip_enabled else "deaktiviert"
            print(f"[i] Frame-Skipping {state} (jede {infer_every}. Frame)")

    cap.release()
    cv2.destroyAllWindows()
