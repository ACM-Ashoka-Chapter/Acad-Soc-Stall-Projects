# ACM Stall — Subway Surfers Body Controller

Play the **real** Subway Surfers on Poki using your body. One script opens the
game and turns your movements into real arrow-key presses.

## Running it

Double-click **`run.bat`**. It opens the Poki game in a **dedicated Chrome
profile** and shows a small camera preview window in the top-left corner.

1. Player stands **~1.5–2m back**, head-to-knees visible in the preview.
2. Click the preview window, press **C**, stand still with arms down for 2s.
   Status turns green: `LIVE - controlling game`.
3. **Click the game** so it has keyboard focus, then hit play. Move.

### Between players

Press **`N`**. That closes the game, wipes the saved progress, reopens it, and
starts a recalibration — so the next player gets the tutorial from scratch.

> Keyboard focus matters: keys go to whatever window is focused. If nothing
> happens, click the game once.

## Controls

| Movement | Action |
|---|---|
| Lean **or** step left/right | Change lane |
| Hop up | Jump |
| Squat down | Roll |

Lean and step **add together**, so a half-lean plus a half-step counts. A
deliberate lean is enough on its own; a casual sway is ignored on purpose.

Sensitivity is set by `lane_threshold` (**0.42**). Tune it live with `-` and
`=` while someone plays, then press `S` to keep it. Lower = twitchier.

Return to the middle to re-centre. Jump and roll re-arm when you return to
standing height.

## Hotkeys (in the preview window)

| Key | Does |
|---|---|
| `N` | **Next player** — reset the game to the tutorial and recalibrate |
| `C` | Recalibrate only |
| `P` | Pause input |
| `-` `=` | Left/right sensitivity, live. Watch the LEAN / STEP bar |
| `S` | Save the current sensitivity into `config.json` |
| `[` `]` | Darker / brighter camera |
| `E` | Toggle auto-exposure |
| `Q` | Quit |

## Tuning at the stall — `config.json`

| Setting | Meaning |
|---|---|
| `lane_threshold` | How far to lean before a lane change. Lower = more sensitive. |
| `lane_release` | How close to centre you must return to re-arm. Keep below `lane_threshold`. |
| `jump_threshold` / `roll_threshold` | Hop / squat depth needed. Lower = easier. |
| `action_cooldown_s` | Minimum gap between jump/roll presses. |
| `tilt_weight` | How much a planted lean counts. Raise if leaning feels dead. |
| `shift_weight` | How much a sideways step counts. |
| `tracking_grace_s` | How long control is held if you briefly leave frame. |
| `smoothing` | Higher = snappier but jitterier. |
| `camera.exposure` | `-6` default. Raise toward `0` if the room is dark. |

All vertical/lateral thresholds are measured in **torso lengths**, not pixels,
so they hold whether the player is tall, short, or standing at a slightly
different distance.

## Troubleshooting

| Problem | Fix |
|---|---|
| Nothing happens in the game | Click the game window — it needs focus |
| `NO PLAYER` in red | Player too close/far, or hips out of frame — step back |
| Lane changes fire randomly / too twitchy | Press `-` a few times, then `S` |
| Leaning does nothing | Press `=` a few times, then `S` |
| Watch the **LEAN / STEP** bar | The amber notches are the firing points - lean and see how close the dot gets |
| Jump never triggers | Lower `jump_threshold` to ~0.08 |
| Preview says under 22fps (amber) | Press `[` a few times — dim rooms slow the webcam badly |
| Camera won't open | Close Zoom/Teams. Run `cam_test.py` to find the right `camera_index` |

## Fresh player each time

Poki remembers "this player has seen the tutorial" in ordinary web storage. On
every launch, and on every `N`, we delete that storage — but keep the HTTP
cache, so the game does not re-download for each player.

This uses its own `browser-profile/` folder next to the script. **Your normal
Chrome profile, bookmarks and logins are never touched.** Delete
`browser-profile/` any time to start completely clean.

Set `"fresh_player_each_launch": false` in `config.json` to turn the reset off.

## Before the event

- Run `setup.ps1` once on the stall laptop (installs deps, downloads the pose
  model, runs the self-test).
- The pose model is cached locally — **the CV works offline**. Only the Poki
  game needs internet, so confirm the venue WiFi reaches poki.com beforehand.
- Tape a floor marker where players should stand. It saves recalibration.
- **Tilt the laptop screen back** so there is headroom above the player. A big
  jump can otherwise push the shoulders out of the top of the frame.

## Files

- `pose_controller.py` — the whole thing
- `config.json` — all tuning
- `browser_session.py` — fresh-profile launching and progress reset
- `test_logic.py` — headless self-test, no webcam needed
- `test_browser.py` — checks the reset clears progress but keeps the cache
- `cam_test.py` — lists working camera indices
