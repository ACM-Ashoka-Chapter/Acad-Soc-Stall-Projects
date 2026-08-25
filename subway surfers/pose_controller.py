"""
ACM Stall - Subway Surfers body controller.

Reads the webcam, tracks the player pose with MediaPipe, and synthesises real
OS-level arrow-key presses so the actual Subway Surfers (Poki) can be played by
stepping, jumping and crouching.

Run:  python pose_controller.py
Keys (click the preview window first):  C = recalibrate,  P = pause,  Q = quit
"""

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import browser_session

import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "pose_landmarker_lite.task"

# MediaPipe pose landmark indices we care about.
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

WINDOW = "ACM x Subway Surfers - Body Controller"

# Theme (BGR).
CYAN = (255, 214, 0)
LIME = (80, 255, 120)
AMBER = (0, 190, 255)
RED = (80, 80, 255)
WHITE = (255, 255, 255)
DIM = (150, 150, 150)
PANEL = (28, 22, 18)


def load_config():
    with open(ROOT / "config.json", encoding="utf-8") as fh:
        return json.load(fh)


class Camera:
    """Threaded webcam grab.

    The built-in webcam blocks ~33ms per read; doing that inline would stack on
    top of inference and cost us a third of our frame rate. A background thread
    keeps only the newest frame so the main loop never waits.

    Auto-exposure is the other trap: in dim light the driver lengthens exposure
    and silently halves the frame rate, so we pin it by default.
    """

    def __init__(self, cfg):
        c = cfg.get("camera", {})
        self.cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_DSHOW)
        if c.get("force_mjpg", True):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
        self.cap.set(cv2.CAP_PROP_FPS, c.get("target_fps", 30))

        self.exposure = c.get("exposure", -6)
        self.manual = c.get("manual_exposure", True)
        self._apply_exposure()

        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _apply_exposure(self):
        if self.manual:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def nudge_exposure(self, delta):
        self.manual = True
        self.exposure = int(np.clip(self.exposure + delta, -13, 0))
        self._apply_exposure()

    def toggle_auto(self):
        self.manual = not self.manual
        self._apply_exposure()

    def is_open(self):
        return self.cap.isOpened()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


class Calibration:
    """The neutral standing pose, in normalised image coordinates."""

    def __init__(self):
        self.ready = False
        self.center_x = 0.5
        self.base_y = 0.5
        self.torso = 0.25          # shoulder->hip distance, our unit of scale
        self.base_tilt = 0.0       # nobody stands perfectly straight
        self._samples = deque(maxlen=45)
        self._collecting = False
        self._started = 0.0

    def start(self):
        self._samples.clear()
        self._collecting = True
        self._started = time.time()
        self.ready = False

    @property
    def collecting(self):
        return self._collecting

    def remaining(self):
        return max(0.0, 2.0 - (time.time() - self._started))

    def feed(self, cx, cy, torso, tilt=0.0):
        if not self._collecting:
            return
        self._samples.append((cx, cy, torso, tilt))
        if self.remaining() <= 0 and len(self._samples) >= 10:
            arr = np.array(self._samples)
            self.center_x, self.base_y, self.torso, self.base_tilt = arr.mean(axis=0)
            self.torso = max(self.torso, 1e-3)
            self._collecting = False
            self.ready = True

    def adapt(self, cy, rate):
        """Slowly follow the neutral height so drift does not break jumps."""
        self.base_y = (1 - rate) * self.base_y + rate * cy


def lateral_signal(pts, torso):
    """How far the player is committing left/right, in torso units.

    Two different things read as "go left" to a player, and we need both:

      shift - the whole body translates sideways (a step). Taken from the
              midpoint of shoulders AND hips, so a step counts even if the
              upper body lags behind.
      tilt  - the torso leans while the feet stay planted. A lean moves the
              shoulder centre by only ~0.17 torso units, so position alone can
              never detect it; we read the hip->shoulder vector off vertical.

    Summed, so half a lean plus half a step still gets you there.

    Hips sit near the bottom edge on a laptop webcam (knees usually out of
    frame) and go jittery there. When their visibility drops we fall back to a
    head-vs-shoulders lean, which needs only the upper body.
    """
    ls, rs, lh, rh, nose = pts[11], pts[12], pts[23], pts[24], pts[0]
    shoulder_x = (ls.x + rs.x) / 2
    hip_vis = min(lh.visibility, rh.visibility)

    if hip_vis >= 0.5:
        hip_x = (lh.x + rh.x) / 2
        body_x = (shoulder_x + hip_x) / 2
        tilt = (shoulder_x - hip_x) / torso
    else:
        body_x = shoulder_x
        tilt = (nose.x - shoulder_x) / torso if nose.visibility >= 0.5 else 0.0
    return body_x, tilt, hip_vis >= 0.5


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.keys = cfg["keys"]
        self.t = cfg["tuning"]
        self.lane = 0                # -1 left, 0 centre, 1 right
        self.last_lane_change = 0.0
        self.last_action = 0.0
        self.airborne = False
        self.rolling = False
        self.enabled = False         # armed only after calibration
        self.paused = False
        self.log = deque(maxlen=6)

    def adjust_sensitivity(self, direction):
        """Nudge the lane threshold live so it can be dialled in at the stall.

        Release tracks the threshold at a fixed ratio, otherwise raising one
        without the other either kills the re-centre or makes it chatter.
        """
        step = 0.03
        self.t = dict(self.t)
        self.t["lane_threshold"] = float(
            np.clip(self.t["lane_threshold"] - direction * step, 0.15, 0.90))
        self.t["lane_release"] = round(self.t["lane_threshold"] * 0.5, 3)

    def _tap(self, action):
        key = self.keys.get(action)
        if not key:
            return
        pydirectinput.press(key)
        self.log.appendleft((time.time(), action.upper()))

    def update(self, offset, lift):
        """offset: lateral position in torso-widths. lift: height above neutral."""
        if not self.enabled or self.paused:
            return
        now = time.time()

        # Lanes: edge-triggered, with hysteresis so it does not chatter.
        target = self.lane
        if offset <= -self.t["lane_threshold"]:
            target = -1
        elif offset >= self.t["lane_threshold"]:
            target = 1
        elif abs(offset) < self.t["lane_release"]:
            target = 0

        if target != self.lane and now - self.last_lane_change > self.t["lane_cooldown_s"]:
            step = 1 if target > self.lane else -1
            for _ in range(abs(target - self.lane)):
                self._tap("right" if step > 0 else "left")
            self.lane = target
            self.last_lane_change = now

        # Jump / roll: one press per gesture, re-armed on return to neutral.
        if lift >= self.t["jump_threshold"]:
            if not self.airborne and now - self.last_action > self.t["action_cooldown_s"]:
                self._tap("jump")
                self.airborne = True
                self.last_action = now
        elif lift <= -self.t["roll_threshold"]:
            if not self.rolling and now - self.last_action > self.t["action_cooldown_s"]:
                self._tap("roll")
                self.rolling = True
                self.last_action = now
        else:
            self.airborne = False
            self.rolling = False


def draw_hud(frame, ctrl, calib, offset, lift, tracked, fps, cam=None, hips_ok=True,
             reacquiring=False):
    h, w = frame.shape[:2]

    # Lane guide rails.
    if calib.ready:
        span = calib.torso * ctrl.t["lane_threshold"]
        for sign in (-1, 1):
            x = int((calib.center_x + sign * span) * w)
            cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1, cv2.LINE_AA)
        cx = int(calib.center_x * w)
        cv2.line(frame, (cx, 0), (cx, h), (45, 45, 45), 1, cv2.LINE_AA)

    # Top status bar.
    strip = frame[0:34, 0:w]
    bar = np.full_like(strip, PANEL, dtype=np.uint8)
    cv2.addWeighted(bar, 0.75, strip, 0.25, 0, strip)

    if calib.collecting:
        state, colour = f"CALIBRATING  {calib.remaining():.1f}s  - stand still, arms down", AMBER
    elif not calib.ready:
        state, colour = "PRESS  C  TO CALIBRATE", AMBER
    elif ctrl.paused:
        state, colour = "PAUSED  - press P to resume", RED
    elif reacquiring:
        state, colour = "REACQUIRING - hold on", AMBER
    elif not tracked:
        state, colour = "NO PLAYER - step back into frame", RED
    else:
        state, colour = "LIVE - controlling game", LIME
    cv2.putText(frame, state, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f}fps", (w - 78, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                DIM if fps > 22 else AMBER, 1, cv2.LINE_AA)
    if cam is not None:
        exp = f"exp {cam.exposure}" if cam.manual else "exp auto"
        cv2.putText(frame, exp, (w - 160, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

    # Lane indicator.
    labels = ("LEFT", "CENTRE", "RIGHT")
    box_w = 74
    for i, label in enumerate(labels):
        x0 = 10 + i * (box_w + 6)
        active = (i - 1) == ctrl.lane and calib.ready
        cv2.rectangle(frame, (x0, h - 40), (x0 + box_w, h - 14),
                      CYAN if active else (55, 55, 55), -1 if active else 1, cv2.LINE_AA)
        cv2.putText(frame, label, (x0 + 8, h - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (20, 20, 20) if active else DIM, 1, cv2.LINE_AA)

    # Vertical gesture meter.
    mx = w - 34
    cv2.rectangle(frame, (mx, 60), (mx + 16, h - 60), (55, 55, 55), 1)
    mid = (60 + h - 60) // 2
    cv2.line(frame, (mx, mid), (mx + 16, mid), DIM, 1)
    fill = int(np.clip(lift / 0.35, -1, 1) * ((h - 120) / 2))
    if fill:
        cv2.rectangle(frame, (mx + 2, mid), (mx + 14, mid - fill),
                      LIME if fill > 0 else AMBER, -1)
    cv2.putText(frame, "JUMP", (mx - 46, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "ROLL", (mx - 46, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)

    # Horizontal lean/step meter. The notches are the lane thresholds, so you
    # can see exactly how close a movement came to firing.
    by = h - 56
    cv2.rectangle(frame, (110, by), (w - 110, by + 12), (55, 55, 55), 1)
    span = (w - 220) / 2
    mid = 110 + span
    thr = ctrl.t["lane_threshold"]
    for sign in (-1, 1):
        nx = int(mid + sign * span)
        cv2.line(frame, (nx, by - 3), (nx, by + 15), AMBER, 1)
    val = float(np.clip(offset / max(thr, 1e-3), -1.35, 1.35))
    px = int(mid + val * span)
    hot = abs(val) >= 1.0
    cv2.rectangle(frame, (int(mid), by + 2), (px, by + 10),
                  LIME if hot else CYAN, -1)
    cv2.circle(frame, (px, by + 6), 5, WHITE if hot else CYAN, -1, cv2.LINE_AA)
    cv2.putText(frame, f"LEAN / STEP   sens {thr:.2f}", (110, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, DIM, 1, cv2.LINE_AA)
    if not hips_ok:
        cv2.putText(frame, "hips out of frame - using head lean", (110, by + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, AMBER, 1, cv2.LINE_AA)

    # Recent inputs, fading out.
    now = time.time()
    for i, (ts, action) in enumerate(list(ctrl.log)[:4]):
        age = now - ts
        if age > 1.6:
            continue
        shade = int(255 * (1 - age / 1.6))
        cv2.putText(frame, action, (10, 60 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (shade, shade, shade), 1, cv2.LINE_AA)


def draw_skeleton(frame, pts):
    h, w = frame.shape[:2]
    bones = [(11, 12), (11, 23), (12, 24), (23, 24),
             (11, 13), (13, 15), (12, 14), (14, 16),
             (23, 25), (25, 27), (24, 26), (26, 28)]
    for a, b in bones:
        if a < len(pts) and b < len(pts):
            pa = (int(pts[a].x * w), int(pts[a].y * h))
            pb = (int(pts[b].x * w), int(pts[b].y * h))
            cv2.line(frame, pa, pb, CYAN, 2, cv2.LINE_AA)
    for idx in (11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 27, 28):
        if idx < len(pts):
            cv2.circle(frame, (int(pts[idx].x * w), int(pts[idx].y * h)), 4, WHITE, -1, cv2.LINE_AA)


def main():
    cfg = load_config()
    if not MODEL.exists():
        sys.exit(f"Missing model file: {MODEL}\nRun setup.ps1 to download it.")

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cam = Camera(cfg)
    if not cam.is_open():
        sys.exit(f"Could not open camera index {cfg['camera_index']}. "
                 "Close other apps using the webcam, or change camera_index in config.json.")

    calib = Calibration()
    ctrl = Controller(cfg)
    smooth_x = smooth_y = smooth_tilt = None
    alpha = cfg["tuning"]["smoothing"]
    fps, last_t = 0.0, time.time()
    frame_idx = 0
    last_good = 0.0
    held_offset = held_lift = 0.0
    reacquiring = False
    grace = cfg["tuning"].get("tracking_grace_s", 0.45)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    pv = cfg["preview"]
    cv2.resizeWindow(WINDOW, pv["width"], int(pv["width"] * cfg["frame_height"] / cfg["frame_width"]))
    cv2.moveWindow(WINDOW, pv["x"], pv["y"])
    if pv["always_on_top"]:
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    print(f"Opening {cfg['game_url']} ...")
    browser_session.launch(cfg["game_url"], fresh=cfg.get("fresh_player_each_launch", True))
    print("Preview window is open. Press C in it to calibrate, then click the game and play.")
    print("Hotkeys: N next player | C recalibrate | P pause | - = sensitivity")
    print("         [ ] exposure | E auto-exposure | S save tuning | Q quit")

    while True:
        frame = cam.read()
        if frame is None:
            cv2.waitKey(10)
            continue
        if cfg["mirror"]:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            frame_idx * 33,
        )

        tracked = False
        offset = lift = 0.0
        hips_ok = True
        if result.pose_landmarks:
            pts = result.pose_landmarks[0]
            ls, rs, lh, rh = pts[L_SHOULDER], pts[R_SHOULDER], pts[L_HIP], pts[R_HIP]
            # Only shoulders are mandatory. On a laptop webcam the hips often
            # sit at the very bottom of the frame, and demanding them made the
            # whole controller drop out.
            vis = min(ls.visibility, rs.visibility)
            if vis >= ctrl.t["min_visibility"]:
                tracked = True
                draw_skeleton(frame, pts)

                cy = (ls.y + rs.y) / 2
                if min(lh.visibility, rh.visibility) >= 0.5:
                    torso = max(abs((lh.y + rh.y) / 2 - cy), 1e-3)
                else:
                    # Fall back to shoulder width, which tracks body scale
                    # about as well and needs nothing below the chest.
                    torso = max(abs(ls.x - rs.x) * 0.9, 1e-3)

                body_x, tilt, hips_ok = lateral_signal(pts, torso)

                smooth_x = body_x if smooth_x is None else (1 - alpha) * smooth_x + alpha * body_x
                smooth_y = cy if smooth_y is None else (1 - alpha) * smooth_y + alpha * cy
                smooth_tilt = tilt if smooth_tilt is None else (1 - alpha) * smooth_tilt + alpha * tilt

                calib.feed(smooth_x, smooth_y, torso, smooth_tilt)

                if calib.ready:
                    # Scale by torso length so distance-to-camera does not matter.
                    shift = (smooth_x - calib.center_x) / calib.torso
                    lean = smooth_tilt - calib.base_tilt
                    offset = (ctrl.t["shift_weight"] * shift
                              + ctrl.t["tilt_weight"] * lean)
                    lift = (calib.base_y - smooth_y) / calib.torso
                    if abs(lift) < 0.04:
                        calib.adapt(smooth_y, ctrl.t["baseline_adapt"])
                    ctrl.enabled = True
                    ctrl.update(offset, lift)

        # A big hop can push the shoulders above the top of the frame. Dropping
        # control the instant that happens would disarm us mid-jump, exactly
        # when the player is moving most. Instead we hold the last good reading
        # briefly: the gesture logic is edge-triggered, so freezing the inputs
        # emits nothing new and simply rides out the gap.
        if tracked:
            last_good = time.time()
            held_offset, held_lift = offset, lift
            reacquiring = False
        else:
            gap = time.time() - last_good
            if gap < grace and calib.ready:
                offset, lift = held_offset, held_lift
                reacquiring = True
            else:
                ctrl.enabled = False
                reacquiring = False

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
        last_t = now

        draw_hud(frame, ctrl, calib, offset, lift, tracked, fps, cam, hips_ok, reacquiring)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            calib.start()
            ctrl.lane = 0
        if key == ord("p"):
            ctrl.paused = not ctrl.paused
        if key == ord("["):
            cam.nudge_exposure(-1)          # darker, faster shutter
        if key == ord("]"):
            cam.nudge_exposure(1)           # brighter, but can cost frame rate
        if key == ord("e"):
            cam.toggle_auto()
        if key == ord("n"):
            # Next player: wipe the game progress and hand them the tutorial.
            browser_session.launch(cfg["game_url"], fresh=True)
            calib.start()
            ctrl.lane = 0
        if key in (ord("-"), ord("_")):
            ctrl.adjust_sensitivity(-1)     # needs a bigger lean
        if key in (ord("="), ord("+")):
            ctrl.adjust_sensitivity(+1)     # needs a smaller lean
        if key == ord("s"):
            cfg["tuning"] = ctrl.t
            with open(ROOT / "config.json", "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"Saved lane_threshold={ctrl.t['lane_threshold']:.2f}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()
    browser_session.close()


if __name__ == "__main__":
    main()
