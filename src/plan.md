Deep code analysis

Read [](file:///c%3A/UofCanada/Winter26/2220/Bottle_Inspection/src/pc_server_final.py)

Read [](file:///c%3A/UofCanada/Winter26/2220/Bottle_Inspection/src/pi_sender_final.py)

Read [](file:///c%3A/UofCanada/Winter26/2220/Bottle_Inspection/requirements.txt)

Read [](file:///c%3A/UofCanada/Winter26/2220/Bottle_Inspection/README.md)

Read [](file:///c%3A/UofCanada/Winter26/2220/Bottle_Inspection/requirements.txt)

Read [](file:///c%3A/UofCanada/Winter26/2220/Bottle_Inspection/README.md)

Read [](file:///c%3A/Users/extra/AppData/Roaming/Code/User/workspaceStorage/a06e2f302469d96a286d0849526227eb/GitHub.copilot-chat/chat-session-resources/f1052eee-7848-460f-ad95-6ee8119e48aa/toolu_bdrk_011Hf9e5JScNRpvomnwLEG7f__vscode-1778747673858/content.txt#1-1), lines 1 to 300

Read [](file:///c%3A/Users/extra/AppData/Roaming/Code/User/workspaceStorage/a06e2f302469d96a286d0849526227eb/GitHub.copilot-chat/chat-session-resources/f1052eee-7848-460f-ad95-6ee8119e48aa/toolu_bdrk_011Hf9e5JScNRpvomnwLEG7f__vscode-1778747673858/content.txt#300-300), lines 300 to 600

Read [](file:///c%3A/Users/extra/AppData/Roaming/Code/User/workspaceStorage/a06e2f302469d96a286d0849526227eb/GitHub.copilot-chat/chat-session-resources/f1052eee-7848-460f-ad95-6ee8119e48aa/toolu_bdrk_011Hf9e5JScNRpvomnwLEG7f__vscode-1778747673858/content.txt#600-600), lines 600 to 800

Now I have everything I need. Let me compile the plan. 

Created memory file [](file:///c%3A/Users/extra/AppData/Roaming/Code/User/workspaceStorage/a06e2f302469d96a286d0849526227eb/GitHub.copilot-chat/memory-tool/memories/ZjEwNTJlZWUtNzg0OC00NjBmLWFkOTUtNmVlODExOWU0OGFh/plan.md)

## Plan: Bottle Inspection – Full Correctness Fix

Each bug is closed by name. No gap between them.

---

### TL;DR
12 distinct bugs cause false counts, wrong classifications, and unsafe relay timing. Four phases close every path that lets a bottle be counted more than once or get the wrong label.

---

### Phase 1 — Frame Pipeline (pc_server_final.py)

**P1-1 — Replace `latest_frame` shared var with `queue.Queue(maxsize=1)`**
The main loop currently re-processes the same frame dozens of times before the capture thread delivers a new one. A `Queue(maxsize=1)` with `put_nowait` ensures: (a) every frame is consumed exactly once, (b) if the main loop falls behind, it automatically discards the stale frame and gets the newest one.
- Replaces `latest_frame`, `frame_lock`, the `with frame_lock` block in the main loop, and the lock-write in the capture thread.
- Main loop reads via `frame_queue.get(timeout=1.0)` inside `try/except queue.Empty`.

**P1-2 — Fix capture thread infinite-loop on disconnect**
`if frame is None: continue` spins forever when the socket closes. Change to `break` and set `running_capture = False` before breaking.

---

### Phase 2 — Tracker Configuration (pc_server_final.py)

**P2-1 — `n_init` 2 → 4**
A track becomes "confirmed" after only 2 frames today. Raising to 4 prevents partial, entering-edge, or misdetected frames from entering the classification pipeline.

**P2-2 — Confidence threshold 0.35 → 0.50**
Low-confidence detections inject noisy labels into track history, causing Good↔No_label confusion. Change in the `model(frame, conf=0.35)` call.

**P2-3 — Base `raw_count` on confirmed tracks, not raw YOLO boxes**
`raw_count = len(detections)` counts raw boxes including overlapping/partial hits. Replace with `raw_count = sum(1 for t in tracks if t.is_confirmed())` (moved to after `tracker.update_tracks`). Eliminates phantom double-counts in the `COUNT:X` message.

---

### Phase 3 — Classification Gate (pc_server_final.py)

This is the core fix for your main complaint. Five interlocking mechanisms close all double-count paths.

**P3-1 — Five new constants**

| Constant | Value | Purpose |
|---|---|---|
| `TRIGGER_X_FRAC` | `0.70` | Fraction of frame width; bottle must cross this line before being counted |
| `MIN_HISTORY_FRAMES` | `8` | Minimum frames of history before a classification commits |
| `MIN_VOTE_FRAC` | `0.60` | Winner must hold ≥60% of votes; ties defer until more history |
| `SPATIAL_RADIUS_PX` | `80` | Pixel radius for spatial dedup |
| `SPATIAL_COOLDOWN_S` | `3.0` | Seconds a counted position stays "hot" |
| `TRACK_CLEANUP_AGE` | `60` | Frames before stale track state is purged from dicts |

**P3-2 — Four new state variables**
- `track_last_cx: dict` — last known center-X per track_id, for crossing detection
- `track_triggered: set` — tracks that have crossed `TRIGGER_X`
- `recently_counted: list` of `(cx, cy, timestamp)` tuples — spatial-temporal dedup log
- `track_last_seen: dict` — track_id → frame_count for cleanup
- `frame_count: int` — increments each iteration

**P3-3 — Extend history window 6 → 15 frames**
Larger vote pool; single noisy frame has far less influence.

**P3-4 — Null guard on `track.get_det_class()`**
Returns `None` when a track is coasting (no detection matched). Add `if label is None: continue` before appending to history.

**P3-5 — Replace the classification trigger with a 4-layer gate**
The current `if len(history) >= 4 and track_id not in sent_ids` block is replaced with all four checks in order:

1. **Crossing-line gate** — only arms a track when its center-X crosses `TRIGGER_X_FRAC * frame_width` left-to-right (`prev_cx < trigger_px <= cx`). Stored in `track_triggered`. A bottle that moves but never crosses the line is never counted.

2. **History depth gate** — `len(history) >= MIN_HISTORY_FRAMES`. If the bottle crosses the line but hasn't accumulated 8 frames of detections yet, defer (it will be checked again next frame).

3. **Dominance gate** — `win_count / len(history) >= MIN_VOTE_FRAC`. If no label has a clear 60% majority, defer. This directly prevents the Good/No_label confusion where 3-vs-3 splits led to arbitrary results.

4. **Spatial-temporal dedup gate** — before committing, clean `recently_counted` of entries older than `SPATIAL_COOLDOWN_S × 2`, then check if any remaining entry is within `SPATIAL_RADIUS_PX` pixels and within `SPATIAL_COOLDOWN_S` seconds. If yes, skip. This is the backstop that catches the case where DeepSort loses a track and assigns a **new track_id** to the same physical bottle that already crossed the line — `sent_ids` wouldn't catch it, but the spatial position would.

Only when all four pass: append to `recently_counted`, add to `sent_ids`, increment counter, send `ID:{track_id}|{winner}`.

**P3-6 — Stale state cleanup each iteration**
At end of loop: increment `frame_count`. Find track_ids in `track_last_seen` not updated in the last `TRACK_CLEANUP_AGE` frames. Pop them from `track_history`, `track_last_cx`, `track_last_seen`. **Do not** remove from `sent_ids` or `track_triggered` — those must persist to block re-counts after cleanup.

---

### Phase 4 — Pi-Side Hardening (pi_sender_final.py)

**P4-1 — Replace `last_processed_id` with `processed_ids: set`**
`last_processed_id` only blocks the immediately previous message. If the PC ever re-sends an older ID (e.g., delayed packet), it slips through and double-counts. A persistent set blocks it permanently for the session.

**P4-2 — Replace polled relay timer with `threading.Timer`**
`classify_off_time` is only checked at the top of the main loop, which can be blocked up to 30 s on `recv_msg`. Replace with:
- `relay_lock = threading.Lock()` declared at module level
- `_relay_off()` callback: acquires lock → sets GPIO17 OFF → logs → releases
- On reject: cancel any existing timer, start `threading.Timer(REJECT_PULSE_SEC, _relay_off)`, store handle in `classify_off_timer`
- Wrap `GPIO.output(RELAY_CLASSIFY, RELAY_ON)` in the same lock
- Remove `classify_off_time` and its polling block entirely

**P4-3 — Clamp count in COUNT handler**
Add `count = min(count, 3)` after parsing to prevent a malformed `COUNT:5` from corrupting the 2-bit relay state.

---

### Relevant Files
- pc_server_final.py — Phases 1, 2, 3
- pi_sender_final.py — Phase 4

---

### Verification
1. **Static bottle test** — hold a bottle still for 10 s; count must increment by exactly 1
2. **Re-entry test** — push bottle past trigger line, pull back left, push right again; spatial dedup must block the second count
3. **Classification accuracy** — run 5 of each class; PC display and Pi totals must match and be correct
4. **Disconnect recovery** — kill Pi mid-run; PC must exit cleanly, not hang in infinite loop
5. **Rapid reject test** — two defective bottles in quick succession; relay pulse fires correctly for each (timer cancel/restart)
6. **Relay bit check** — log and verify GPIO27/22 binary pattern for counts 0, 1, 2, 3
7. **No stale-frame test** — print `frame_count` on overlay; it must never show the same value on two consecutive rendered frames

---

### Decisions
- `TRIGGER_X_FRAC = 0.70` assumes left-to-right conveyor. If reversed, the crossing condition flips to `prev_cx > trigger_px and cx <= trigger_px`.
- `SPATIAL_RADIUS_PX = 80` matches `SINGLE_BOTTLE_PX`. Tune down if bottles are packed closer.
- `MIN_HISTORY_FRAMES = 8` at ~15 fps over network ≈ 0.5 s of observation. Reduce to 6 if the conveyor speed is high and bottles are on-screen briefly.
- Running-total counters (`good_count` etc.) intentionally persist for the session — no change to this behaviour.

---

Ready to implement? I can do all four phases at once or one at a time.