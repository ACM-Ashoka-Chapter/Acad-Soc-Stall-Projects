import sys, types, pathlib, shutil
stub = types.ModuleType("pydirectinput"); stub.PAUSE=0; stub.FAILSAFE=False; stub.press=lambda k: None
sys.modules["pydirectinput"] = stub
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import browser_session as bs
import pose_controller as pc

fails = []
def check(n, c):
    print(("  PASS  " if c else "  FAIL  ") + n)
    if not c: fails.append(n)

print("\n[fresh-player storage reset]")
tmp = pathlib.Path("_proftest")
if tmp.exists(): shutil.rmtree(tmp)
d = tmp / "Default"
for name in ["Local Storage", "IndexedDB", "Session Storage", "Service Worker"]:
    (d / name).mkdir(parents=True)
    (d / name / "poki.leveldb").write_text("tutorial_seen=true")
(d / "Cookies").write_text("poki-session")
for name in ["Cache", "Code Cache"]:
    (d / name).mkdir(parents=True)
    (d / name / "asset.bin").write_bytes(b"x" * 5000)
(d / "Preferences").write_text("{}")

cleared, failed = bs.clear_site_storage(tmp)
check(f"cleared storage areas ({cleared})", cleared >= 5)
check("no failures", failed == 0)
check("Local Storage gone", not (d / "Local Storage").exists())
check("IndexedDB gone", not (d / "IndexedDB").exists())
check("Service Worker gone", not (d / "Service Worker").exists())
check("Cookies gone", not (d / "Cookies").exists())
check("asset Cache PRESERVED", (d / "Cache" / "asset.bin").exists())
check("Code Cache PRESERVED", (d / "Code Cache").exists())
check("Preferences preserved", (d / "Preferences").exists())

# Idempotent: running twice must not blow up.
c2, f2 = bs.clear_site_storage(tmp)
check("safe to run on already-clean profile", f2 == 0)
shutil.rmtree(tmp)

print("\n[profile isolation]")
real = pathlib.Path.home() / "AppData/Local/Google/Chrome/User Data"
check("uses its own profile dir, not the real one",
      bs.PROFILE.resolve() != real.resolve() and "Downloads" in str(bs.PROFILE.resolve()))
check("real Chrome profile still intact", (real / "Default").exists())
check("finds a browser", bs.find_browser() is not None)

print("\n[live sensitivity]")
cfg = pc.load_config()
ctrl = pc.Controller(cfg)
start = ctrl.t["lane_threshold"]
ctrl.adjust_sensitivity(-1)
check("minus makes it LESS sensitive", ctrl.t["lane_threshold"] > start)
ctrl.adjust_sensitivity(+1); ctrl.adjust_sensitivity(+1)
check("plus makes it MORE sensitive", ctrl.t["lane_threshold"] < start)
check("release stays below threshold",
      ctrl.t["lane_release"] < ctrl.t["lane_threshold"])
for _ in range(60): ctrl.adjust_sensitivity(+1)
check("clamps at sensitive end", ctrl.t["lane_threshold"] >= 0.15)
for _ in range(120): ctrl.adjust_sensitivity(-1)
check("clamps at insensitive end", ctrl.t["lane_threshold"] <= 0.90)
check("does not mutate shared config dict",
      cfg["tuning"]["lane_threshold"] == start)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
