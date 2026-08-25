"""Headless checks: gesture state machine + MediaPipe model load. No webcam needed."""
import sys, types, time, json, pathlib

# Stub pydirectinput so nothing is actually typed while testing.
sent = []
stub = types.ModuleType("pydirectinput")
stub.PAUSE = 0
stub.FAILSAFE = False
stub.press = lambda k: sent.append(k)
sys.modules["pydirectinput"] = stub

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pose_controller as pc

cfg = pc.load_config()
fails = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        fails.append(name)


def fresh():
    sent.clear()
    c = pc.Controller(cfg)
    c.enabled = True
    c.t = dict(c.t, action_cooldown_s=0.0, lane_cooldown_s=0.0)
    return c

print("\n[gesture state machine]")

c = fresh()
c.update(-0.8, 0)
check("step left -> one LEFT press", sent == ["left"])

c.update(-0.8, 0); c.update(-0.9, 0)
check("holding left does not repeat", sent == ["left"])

c.update(0.0, 0)
check("return to centre -> RIGHT to recentre", sent == ["left", "right"])

c = fresh()
c.update(0.9, 0)
c.update(-0.9, 0)
check("left-to-right across centre -> two presses", sent == ["right", "left", "left"])

c = fresh()
c.update(cfg["tuning"]["lane_release"] * 0.5, 0)
check("inside deadzone -> no press", sent == [])

c = fresh()
c.update(0, 0.2)
c.update(0, 0.25)
check("jump fires once while airborne", sent == ["up"])
c.update(0, 0.0)
c.update(0, 0.2)
check("jump re-arms after landing", sent == ["up", "up"])

c = fresh()
c.update(0, -0.3)
c.update(0, -0.35)
check("crouch fires once", sent == ["down"])

c = fresh()
c.paused = True
c.update(-0.9, 0.3)
check("paused sends nothing", sent == [])

c = fresh()
c.enabled = False
c.update(-0.9, 0.3)
check("untracked sends nothing", sent == [])

print("\n[calibration]")
cal = pc.Calibration()
check("starts uncalibrated", not cal.ready)
cal.start()
cal._started = time.time() - 3          # pretend the 2s window elapsed
for _ in range(12):
    cal.feed(0.5, 0.4, 0.22)
check("calibrates after enough samples", cal.ready)
check("captures centre", abs(cal.center_x - 0.5) < 1e-6)
check("captures torso scale", abs(cal.torso - 0.22) < 1e-6)
base = cal.base_y
cal.adapt(0.45, 0.02)
check("baseline drifts toward new height", base < cal.base_y < 0.45)

print("\n[lateral signal: lean vs step]")


class LM:
    def __init__(self, x, y, v=1.0):
        self.x, self.y, self.visibility = x, y, v


def body(shoulder_x, hip_x, hip_vis=1.0, nose_x=None):
    """Synthetic 33-landmark pose at a plausible stall distance."""
    pts = [LM(0.5, 0.1) for _ in range(33)]
    half = 0.09
    pts[11] = LM(shoulder_x - half, 0.40)
    pts[12] = LM(shoulder_x + half, 0.40)
    pts[23] = LM(hip_x - 0.06, 0.62, hip_vis)
    pts[24] = LM(hip_x + 0.06, 0.62, hip_vis)
    pts[0] = LM(shoulder_x if nose_x is None else nose_x, 0.22)
    return pts


T = cfg["tuning"]
THR = T["lane_threshold"]


def combined(pts, torso=0.22, base_tilt=0.0, center=0.5):
    bx, tilt, ok = pc.lateral_signal(pts, torso)
    val = T["shift_weight"] * ((bx - center) / torso) + T["tilt_weight"] * (tilt - base_tilt)
    return val, ok


v, _ = combined(body(0.5, 0.5))
check("standing straight -> near zero", abs(v) < 0.05)

# A planted lean: shoulders move, hips stay. Position alone could never see
# this - it is the case that was completely broken before the tilt term.
def lean(units):
    """Lean of `units` torso-lengths at the shoulders, feet planted."""
    return body(0.5 + units * 0.22, 0.5)

v, _ = combined(lean(0.20))
check("deliberate lean fires (%.2f >= %s)" % (v, THR), v >= THR)

v, _ = combined(lean(-0.20))
check("lean is signed the other way too", v <= -THR)

# A committed 20cm step at ~1.8m, whole body translating.
step = 0.20 / 2.08
v, _ = combined(body(0.5 + step, 0.5 + step))
check("committed 20cm step fires (%.2f >= %s)" % (v, THR), v >= THR)

# The point of raising the threshold: a casual half-lean must NOT fire, and
# must fall inside the release band so it does not leave a lane latched.
v, _ = combined(lean(0.08))
check("casual half-lean stays quiet (%.2f)" % v, abs(v) < THR)

v, _ = combined(body(0.5 + 0.012, 0.5 + 0.008))
check("small sway inside release band (%.2f)" % v, abs(v) < T["lane_release"])

# Guard the relationship itself, not just today numbers.
check("release band is below the fire threshold", T["lane_release"] < THR)

# Hips out of frame -> head-lean fallback.
bx, tilt, ok = pc.lateral_signal(body(0.5, 0.5, hip_vis=0.1, nose_x=0.55), 0.22)
check("detects hips missing", not ok)
check("falls back to head lean", tilt > 0.15)
bx, tilt, ok = pc.lateral_signal(body(0.5, 0.5, hip_vis=0.1), 0.22)
check("upright head -> no false lean", abs(tilt) < 0.02)

print("\n[calibrated neutral tilt]")
cal2 = pc.Calibration()
cal2.start()
cal2._started = time.time() - 3
for _ in range(12):
    cal2.feed(0.5, 0.4, 0.22, 0.08)          # a habitual 0.08 resting lean
check("captures resting tilt", abs(cal2.base_tilt - 0.08) < 1e-6)
v, _ = combined(body(0.5, 0.5), base_tilt=cal2.base_tilt)
check("resting lean is subtracted out", abs(v + T["tilt_weight"] * 0.08) < 1e-6)

print("\n[mediapipe model]")
try:
    import numpy as np, mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(pc.MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.zeros((480, 640, 3), dtype=np.uint8))
    r = lm.detect_for_video(img, 33)
    lm.close()
    check("model loads and runs a frame", r is not None)
except Exception as e:
    check(f"model loads and runs a frame ({type(e).__name__}: {e})", False)

print("\n[hud rendering]")
try:
    import numpy as np
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    cc = pc.Calibration(); cc.ready = True; cc.center_x = 0.5; cc.base_y = 0.4; cc.torso = 0.22
    pc.draw_hud(f, fresh(), cc, 0.3, 0.1, True, 29.7, None, True)
    pc.draw_hud(f, fresh(), cc, -0.9, -0.2, True, 29.7, None, False)
    check("draw_hud runs without error", f.any())
except Exception as e:
    check(f"draw_hud runs without error ({type(e).__name__}: {e})", False)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
