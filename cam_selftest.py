import time, argparse, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--seconds", type=int, default=5)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--backend", choices=["auto","dshow","msmf"], default="auto")
args = ap.parse_args()

backend_flag = 0
if args.backend == "dshow":
    backend_flag = cv2.CAP_DSHOW
elif args.backend == "msmf":
    backend_flag = cv2.CAP_MSMF

cap = cv2.VideoCapture(args.camera, backend_flag) if backend_flag else cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Kamera {args.camera} nicht verfügbar (Backend: {args.backend}).")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

frames = 0; t0 = time.time()
while time.time() - t0 < args.seconds:
    ok, frame = cap.read()
    if not ok: break
    frames += 1
    cv2.imshow(f"Camera {args.camera} ({args.backend}) - ESC endet", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()
dur = max(time.time()-t0, 1e-6)
print(f"[OK] {frames} Frames in {dur:.2f}s -> ~{frames/dur:.1f} FPS")