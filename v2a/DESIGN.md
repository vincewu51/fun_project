# Video-to-Action (V2A) Pipeline — Architecture & Implementation Plan

> **Goal:** Convert any RGB video of a human hand/arm demo into a LeRobot-compatible training
> dataset for the SO101 follower arm — no physical robot required during recording.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          V2A Pipeline                               │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────┐ │
│  │  Video   │───▶│  Pose Stage  │───▶│  Retarget    │───▶│  IK   │ │
│  │  Input   │    │  (MediaPipe  │    │  Stage       │    │ Stage │ │
│  │  (MP4)   │    │   + HaMeR)   │    │  (affine +   │    │(ikpy) │ │
│  └──────────┘    └──────────────┘    │   normalize) │    └───┬───┘ │
│                        │             └──────────────┘        │     │
│                        │                                      │     │
│                  3D keypoints                          joint angles  │
│                  per frame                             per frame     │
│                  (21 landmarks                                       │
│                   + wrist pose)                                      │
│                                                               │     │
│                                                        ┌──────▼───┐ │
│                                                        │ LeRobot  │ │
│                                                        │ Dataset  │ │
│                                                        │ Writer   │ │
│                                                        └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Module interfaces (Python)

```
VideoLoader         → Iterator[Frame]          # (H, W, 3) uint8 + timestamp
PoseEstimator       → Iterator[HandPose]       # wrist SE3 + 21 keypoints, per frame
RetargetingMapper   → Iterator[EEPose]         # (x, y, z, rx, ry, rz) in robot base frame
IKSolver            → Iterator[JointAngles]    # (6,) float32 in degrees, SO101 convention
GripperEstimator    → Iterator[float]          # [0,1] normalized open→close
LeRobotWriter       → LeRobotDataset           # HF dataset on disk
```

---

## 2. Technology Recommendations

### Stage 1 — Pose Estimation

**Use: MediaPipe Hands (primary) + HaMeR (fallback/upgrade)**

| Library | Pros | Cons |
|---------|------|------|
| **MediaPipe Hands** | Real-time, 21 3D landmarks, robust, easy pip install | Wrist rotation is relative, no absolute depth |
| **HaMeR** | Full 3D MANO mesh + absolute wrist SE3 | Heavier model, setup cost, needs GPU |
| FoundPose | Object-level, not hand-specific | Wrong tool |

**Recommendation:** Start with MediaPipe for iteration speed. The wrist Z depth from MediaPipe is
scale-normalized (relative to hand size) — acceptable once we calibrate the affine map against D1
data. Upgrade to HaMeR when you need better absolute wrist orientation for tasks with significant
wrist rotation (e.g., screwing, pouring).

**Gripper estimation:** Use the normalized distance between the thumb tip (landmark 4) and index
finger tip (landmark 8). This correlates well with gripper open/close for pick-and-place tasks.

```python
# Gripper proxy: 0 = fully open, 1 = fully closed
pinch_dist = np.linalg.norm(landmarks[4] - landmarks[8])
gripper_norm = 1.0 - np.clip(pinch_dist / MAX_PINCH_DIST, 0, 1)
```

### Stage 2 — Retargeting (Human → Robot EE)

**Use: Affine transform (linear, calibrated per setup)**

A learned mapping is overkill for phase 1. An affine transform trained on D1 paired data is
interpretable, fast, and corrects for:
- Coordinate frame differences (camera vs robot base)
- Scale differences (human arm reach vs SO101 workspace)
- Constant offsets (camera mounting position)

```
x_robot = A @ x_human + b       # A: (3,3), b: (3,)
```

Calibrate by collecting 10–20 paired (hand pose, robot EE pose) samples via teleoperation
with a marker on the wrist.

### Stage 3 — Inverse Kinematics

**Use: ikpy (primary) — you already have FK/IK utilities, extend them**

| Library | Pros | Cons |
|---------|------|------|
| **ikpy** | Pure Python, lightweight, easy SO101 URDF loading | Slower than C++ solvers |
| PyBullet | Physics sim + IK, visualization | Heavy dependency, overkill |
| pinocchio | Fast, full rigid body dynamics | Complex setup, C++ bindings |

**Recommendation:** ikpy with your SO101 URDF. It uses Levenberg-Marquardt by default and handles
6-DOF arms well. Add joint limit clamping after solving. For offline conversion speed is fine.

```python
import ikpy.chain
chain = ikpy.chain.Chain.from_urdf_file("so101.urdf", active_links_mask=[...])
joints = chain.inverse_kinematics(target_position=ee_xyz, target_orientation=ee_rot)
```

If IK diverges on edge cases, fall back to the Jacobian pseudoinverse method in your existing
`FK_IK/utility.py`.

### Stage 4 — LeRobot Dataset Writer

**Use: `LeRobotDataset` directly (v0.3.3, same as your existing converters)**

Follow the exact pattern from `lerobot/isaaclab2lerobot_e.py` — reuse `SINGLE_ARM_FEATURES`.

---

## 3. Coordinate Space Design

### Three coordinate frames

```
Camera frame (C)  →  [Retarget]  →  Robot EE frame (E)  →  [IK]  →  Joint space (J)
```

**Camera frame:** MediaPipe outputs normalized image coordinates + pseudo-depth. Convert to metric
using the known hand size prior (average human palm width ≈ 80mm) to recover rough metric scale.

**Calibration procedure (one-time, per camera mounting):**

1. Collect 20 "calibration frames" via teleoperation: human holds a bright marker at fixed
   robot EE positions while the camera records.
2. For each frame: record `(x_hand_mp, y_hand_mp, z_hand_mp)` from MediaPipe and ground truth
   `(x_robot, y_robot, z_robot)` from robot joint state via FK.
3. Fit affine transform `A, b` using `np.linalg.lstsq`.
4. Save `A, b` to `v2a/calibration/affine_map.npz`.

**Normalization strategy:**

Normalize all trajectories to robot workspace bounds before IK:
```python
x_clamped = np.clip(x_robot, WORKSPACE_MIN, WORKSPACE_MAX)
```

Define workspace bounds from your real teleoperation data statistics (p5–p95 percentile of
collected D1 positions).

**Wrist orientation:** MediaPipe gives wrist orientation as a rotation matrix relative to camera.
Apply the same affine rotation part `A` to rotate the orientation vector. For the gripper axis
(pointing from palm toward fingers), use the vector between wrist (0) and middle-finger-MCP (9).

---

## 4. LeRobot Integration

### Dataset structure

Reuse `SINGLE_ARM_FEATURES` from `lerobot/isaaclab2lerobot_e.py` exactly:

```python
# action: (6,) — [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# observation.state: (6,) — same joint positions
# observation.images.front: (H, W, 3) — the input video frames
```

For V2A datasets, `observation.state` at time `t` = the IK solution at time `t` (since we have no
real robot state, action and observation.state are the same trajectory, time-shifted by 1 frame).

### Writer pattern

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="ywu67/v2a_demo_001",
    fps=30,
    features=SINGLE_ARM_FEATURES,
    robot_type="so101_follower",
)

for episode_idx, episode_frames in enumerate(episodes):
    for frame in episode_frames:
        dataset.add_frame({
            "action": frame.joint_angles,           # (6,) float32
            "observation.state": frame.joint_angles, # (6,) float32
            "observation.images.front": frame.rgb,   # (H, W, 3) uint8
        })
    dataset.save_episode(task=episode_task_description)

dataset.consolidate()
```

### Key correctness rules

- **fps must match video fps** (30 for your setup). Set this at dataset creation.
- **Action at index t = joint angles at t** (not t+1). LeRobot's ACT/DP training handles the
  temporal offset internally.
- **Episode boundaries**: call `save_episode()` once per input video clip. Never mix clips.
- **Gripper convention**: match your teleoperation convention. Check `D1` gripper range and
  normalize V2A gripper output to the same range (degrees, not [0,1]).

---

## 5. Validation Pipeline Design

### Data collection (D1 + D2 simultaneously)

```
Teleoperation session:
  ├── Record: robot joint states at 30fps           → D1 actions
  ├── Record: wrist camera video at 30fps           → D1 images + source for D2
  └── Record: front camera video at 30fps           → D1 + D2 front images

V2A conversion:
  └── Run wrist camera video through V2A pipeline   → D2 actions
```

**Sync**: Use LeRobot's built-in timestamp synchronization. The wrist camera video IS the video
you feed to V2A, so frame alignment is exact (no sync problem).

### Metrics

```python
# 1. Per-timestep MSE on joint trajectories
mse = np.mean((d1_actions - d2_actions) ** 2, axis=0)  # (6,) per-joint

# 2. DTW distance on full episode (handles minor timing offsets)
from dtaidistance import dtw
dtw_dist = dtw.distance(d1_actions.flatten(), d2_actions.flatten())

# 3. Workspace overlap: % of D2 EE positions within D1 workspace convex hull
from scipy.spatial import ConvexHull
hull = ConvexHull(d1_ee_positions)
overlap = np.mean([is_inside_hull(p, hull) for p in d2_ee_positions])

# 4. Policy rollout success rate (gold standard)
# Train ACT on D1 → eval on robot → success rate S1
# Train ACT on D2 → eval on robot → success rate S2
# Target: S2 / S1 > 0.7
```

### File structure

```
v2a/
├── DESIGN.md                      # this file
├── pipeline/
│   ├── __init__.py
│   ├── video_loader.py            # VideoLoader class
│   ├── pose_estimator.py          # MediaPipe wrapper → HandPose dataclass
│   ├── retargeting.py             # AffineRetargeter: fit(), transform()
│   ├── ik_solver.py               # IKSolver wrapping ikpy + SO101 URDF
│   ├── gripper_estimator.py       # pinch-distance → gripper angle
│   └── lerobot_writer.py          # LeRobotWriter using SINGLE_ARM_FEATURES
├── calibration/
│   ├── collect_calibration.py     # script: teleoperate + record paired samples
│   ├── fit_affine.py              # script: fit A, b from calibration data
│   └── affine_map.npz             # saved A, b (generated, not committed)
├── validation/
│   ├── compare_d1_d2.py           # compute MSE, DTW, workspace overlap
│   └── visualize_trajectories.py  # matplotlib: D1 vs D2 joint plots
├── convert_video.py               # CLI entrypoint: video → LeRobot dataset
└── config.py                      # workspace bounds, joint limits, camera params
```

---

## 6. Phased Implementation Roadmap

### Phase 1 — Pose → IK proof of concept (Week 1–2) ★ Start here

**Goal:** Given a single short video, produce a plausible joint angle trajectory. No LeRobot
integration yet. Just verify the pipeline runs end-to-end.

**Deliverables:**
- `pose_estimator.py`: MediaPipe on video, output 21 landmarks per frame as numpy array
- `retargeting.py`: hardcoded (no calibration yet) rough affine using known camera setup
- `ik_solver.py`: ikpy with SO101 URDF, solve for each frame
- `visualize_trajectories.py`: plot joint angles over time, visually sanity-check

**Complexity:** Low. This is pure Python, no robotics hardware needed.

**Key question to answer:** Does the IK produce smooth, physically plausible trajectories?
If joint angles jump wildly → add temporal smoothing (Savitzky-Golay filter, window=5).

---

### Phase 2 — Calibration + LeRobot packaging (Week 3–4)

**Goal:** Calibrate the affine map from real paired data and write a valid LeRobot dataset.

**Deliverables:**
- `collect_calibration.py`: during teleoperation session, record 20 paired wrist-marker poses
- `fit_affine.py`: `np.linalg.lstsq` fit, save `affine_map.npz`
- `lerobot_writer.py`: write IK output → LeRobot dataset (use `isaaclab2lerobot_e.py` as template)
- `convert_video.py`: CLI: `python convert_video.py --video demo.mp4 --output ./dataset`

**Complexity:** Medium. LeRobot dataset API has gotchas (fps, episode boundaries, feature shapes).

---

### Phase 3 — D1/D2 comparison + calibration refinement (Week 5–6)

**Goal:** Quantitatively validate that D2 trajectories approximate D1.

**Deliverables:**
- Collect 5–10 episodes with synchronized D1 ground truth
- `compare_d1_d2.py`: MSE/DTW/workspace overlap report
- Iterate on affine map calibration until DTW distance is within 2× of teleoperation noise floor
- `gripper_estimator.py`: tune pinch threshold to match real gripper open/close events in D1

**Complexity:** Medium. DTW needs careful normalization. Workspace overlap gives fast signal.

---

### Phase 4 — Policy training + real robot evaluation (Week 7–10)

**Goal:** Train ACT on D1 and D2 separately, evaluate success rates on real robot.

**Deliverables:**
- LeRobot training runs: `lerobot-train --dataset.repo_id ywu67/v2a_d1 ...`
- LeRobot eval runs on real SO101 for both M1 and M2
- Final report: success rate comparison, failure mode analysis
- If S2/S1 < 0.7: analyze top failure modes → improve calibration or add HaMeR

**Complexity:** High (real robot eval is always the hardest part). Plan for 2–3 eval iterations.

---

## 7. Key Risks and Mitigations

### Risk 1 — Monocular depth ambiguity (HARDEST PROBLEM)

**Problem:** MediaPipe's Z coordinate is pseudo-depth normalized by hand size, not metric depth.
The Z axis has the lowest signal quality.

**Mitigation:**
- The affine calibration absorbs systematic Z offset/scale errors.
- Use a fixed camera mount (tripod) at a known distance from the robot — this constrains Z range.
- For pick-and-place: Z mostly matters at grasp/release events. Detect these via gripper state
  changes rather than raw Z trajectory.
- If Z errors are severe, consider adding a second camera at 90° and fusing depth estimates.

---

### Risk 2 — Workspace mismatch (human arm ≠ SO101 workspace)

**Problem:** Human arm reach (~70cm) >> SO101 workspace (~30cm). Raw retargeting will send IK
out-of-reach.

**Mitigation:**
- Clip EE targets to SO101 workspace box before IK. Define bounds from D1 statistics.
- The affine scaling factor handles gross scale mismatch.
- Normalize human workspace to unit cube, map to robot workspace unit cube:
  ```python
  x_norm = (x_human - human_center) / human_range
  x_robot = x_norm * robot_range + robot_center
  ```

---

### Risk 3 — IK instability (joint flipping near singularities)

**Problem:** SO101 has wrist singularities. Near these, IK solutions jump discontinuously.

**Mitigation:**
- Warm-start IK from the previous frame's solution: `thetalist0 = prev_joints`
- Add post-IK smoothing: Savitzky-Golay with window=5, poly=2
- Clamp joint velocity: `delta = np.clip(new - prev, -MAX_VEL/fps, MAX_VEL/fps)`
- Define MAX_VEL per joint from D1 statistics (p99 of observed velocities)

---

### Risk 4 — Gripper timing mismatch

**Problem:** Pinch detection threshold may not align with actual grasp events in D1.

**Mitigation:**
- Treat gripper as a binary event (open/close), not continuous.
- In Phase 3, align gripper event timestamps between D1 and D2.
- Tune pinch threshold on calibration episodes where you know exact grasp timing.

---

### Risk 5 — Frame rate / timestamp drift

**Problem:** If video fps ≠ LeRobot fps, temporal alignment breaks.

**Mitigation:**
- Always record at 30fps (matches your lerobot-record setup).
- Use `ffmpeg -r 30` to re-encode if source video has variable fps.
- Store original timestamps in episode metadata for debugging.

---

## Quick-start: first 3 commands

```bash
# 1. Install dependencies
pip install mediapipe ikpy dtaidistance

# 2. Test pose estimator on a short video
python -c "
import mediapipe as mp
import cv2
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture('test.mp4')
ret, frame = cap.read()
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print(results.multi_hand_landmarks)
"

# 3. Run full pipeline (Phase 1 target)
python convert_video.py --video demo.mp4 --output ./output_dataset --dry-run
```
