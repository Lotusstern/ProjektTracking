import time, argparse, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--seconds", type=int, default=5)
args = ap.parse_args()

cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
if not cap.isOpened():
    raise SystemExit(f"Kamera {args.camera} nicht verfügbar (MSMF).")

# MJPG erzwingen – viele Treiber werden so „williger“
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

for (w,h) in [(1280,720),(1024,576),(800,600),(640,480)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    if ok:
        print(f"[OK] Erste Aufnahme @ {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        break
else:
    cap.release()
    raise SystemExit("Keine unterstützte Auflösung gefunden.")

t0=time.time(); frames=0
while time.time()-t0 < args.seconds:
    ok, frame = cap.read()
    if not ok: break
    frames += 1
    cv2.imshow("Selftest (ESC beendet)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()
dur=max(time.time()-t0,1e-6)
print(f"[OK] {frames} Frames in {dur:.2f}s -> ~{frames/dur:.1f} FPS")