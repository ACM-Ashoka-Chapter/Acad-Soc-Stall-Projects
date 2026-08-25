"""Launches the game in a throwaway browser profile so every player is new.

Poki keeps "have you played before" state in ordinary web storage (localStorage
and IndexedDB, on both poki.com and the game frame origin). Clearing that is
what brings the tutorial back.

We deliberately use our OWN profile directory inside this folder and never
touch the player's real Chrome profile - their bookmarks, logins and history
are not involved in any of this.

We also clear only *site storage*, not the HTTP cache. Progress lives in site
storage; the game's several MB of assets live in the cache. Keeping the cache
means the second player is not waiting on venue WiFi to re-download the game.
"""

import shutil
import subprocess
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
PROFILE = ROOT / "browser-profile"

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

# Per-origin storage that carries "this player has played before".
STORAGE = [
    "Local Storage",
    "Session Storage",
    "IndexedDB",
    "Service Worker",
    "databases",
    "Local Extension Settings",
    "Cookies",
    "Cookies-journal",
]

# Worth keeping: these make the reload fast.
KEEP = ["Cache", "Code Cache", "GPUCache", "DawnGraphiteCache", "DawnWebGPUCache"]


def find_browser():
    for exe in BROWSERS:
        if exe.exists():
            return exe
    return None


def clear_site_storage(profile=PROFILE):
    """Delete progress-bearing storage, leave the asset cache alone.

    Returns (cleared, failed) counts. Never raises - a stall demo should still
    start even if one file is locked.
    """
    cleared = failed = 0
    for sub in ("Default", ""):
        base = profile / sub if sub else profile
        if not base.is_dir():
            continue
        for name in STORAGE:
            target = base / name
            if not target.exists():
                continue
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                cleared += 1
            except OSError:
                failed += 1
    return cleared, failed


_proc = None


def close(timeout=6.0):
    """Shut the game browser down and wait for it to release the profile.

    This matters more than it looks. Chrome holds its storage files open, so
    clearing them underneath a running instance either fails or gets rewritten
    from memory on exit. Worse, relaunching with the same --user-data-dir while
    an instance is alive just opens a tab in the OLD session - the reset would
    silently do nothing. So: kill first, then clear, then start clean.
    """
    global _proc
    if _proc is None:
        return
    try:
        _proc.terminate()
        _proc.wait(timeout=timeout)
    except Exception:
        # Chrome spawns a process tree; take the whole thing down.
        try:
            subprocess.run(["taskkill", "/PID", str(_proc.pid), "/T", "/F"],
                           capture_output=True, timeout=timeout)
        except Exception:
            pass
    _proc = None
    time.sleep(0.6)          # let Windows release the file locks


def launch(url, fresh=True):
    """Open the game. Returns the Popen handle, or None if we fell back."""
    global _proc
    exe = find_browser()
    if exe is None:
        print("No Chrome/Edge found - opening default browser "
              "(player progress will NOT reset).")
        webbrowser.open(url)
        return None

    close()

    if fresh:
        PROFILE.mkdir(parents=True, exist_ok=True)
        cleared, failed = clear_site_storage()
        for _ in range(3):
            if not failed:
                break
            time.sleep(0.7)          # a lock we raced; give it another go
            cleared, failed = clear_site_storage()
        note = f"cleared {cleared} storage areas"
        if failed:
            note += f" ({failed} still locked - close stray Chrome windows)"
        print(f"Fresh player: {note}.")

    cmd = [
        str(exe),
        f"--user-data-dir={PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--hide-crash-restore-bubble",
        "--start-maximized",
        "--autoplay-policy=no-user-gesture-required",
        url,
    ]
    try:
        _proc = subprocess.Popen(cmd)
        return _proc
    except OSError as exc:
        print(f"Could not launch {exe.name} ({exc}); falling back.")
        webbrowser.open(url)
        return None


if __name__ == "__main__":
    # Manual check: python browser_session.py
    import json
    cfg = json.load(open(ROOT / "config.json", encoding="utf-8"))
    print("browser:", find_browser())
    launch(cfg["game_url"])
