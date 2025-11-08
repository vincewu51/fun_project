#!/usr/bin/env python3
"""
Script to filter R1Pro proprioception state to only include features allowed in BEHAVIOR Challenge standard track.
Removes: base joints, global position, global orientation
"""

import numpy as np

# R1Pro allowed proprioception indices (standard track)
# Based on OmniGibson/omnigibson/learning/utils/eval_utils.py
# Excluding: base joints (indices 0-5 in joint_qpos), global pos/ori

ALLOWED_R1PRO_INDICES = {
    # Joint positions (excluding first 6 base joints)
    "joint_qpos": np.s_[6:28],  # indices 6-27 from original 0-27
    "joint_qpos_sin": np.s_[34:56],  # indices 34-55 from original 28-55 (skip first 6)
    "joint_qpos_cos": np.s_[62:84],  # indices 62-83 from original 56-83 (skip first 6)

    # Joint velocities and efforts (full, base velocities ARE allowed)
    "joint_qvel": np.s_[84:112],
    "joint_qeffort": np.s_[112:140],

    # SKIP robot_pos (140-142) - global position NOT allowed
    # SKIP robot_ori_cos (143-145) - global orientation NOT allowed
    # SKIP robot_ori_sin (146-148) - global orientation NOT allowed
    # SKIP robot_2d_ori (149) - NOT allowed
    # SKIP robot_2d_ori_cos (150) - NOT allowed
    # SKIP robot_2d_ori_sin (151) - NOT allowed

    # Robot velocities (allowed)
    "robot_lin_vel": np.s_[152:155],
    "robot_ang_vel": np.s_[155:158],

    # Left arm
    "arm_left_qpos": np.s_[158:165],
    "arm_left_qpos_sin": np.s_[165:172],
    "arm_left_qpos_cos": np.s_[172:179],
    "arm_left_qvel": np.s_[179:186],
    "eef_left_pos": np.s_[186:189],
    "eef_left_quat": np.s_[189:193],
    "gripper_left_qpos": np.s_[193:195],
    "gripper_left_qvel": np.s_[195:197],

    # Right arm
    "arm_right_qpos": np.s_[197:204],
    "arm_right_qpos_sin": np.s_[204:211],
    "arm_right_qpos_cos": np.s_[211:218],
    "arm_right_qvel": np.s_[218:225],
    "eef_right_pos": np.s_[225:228],
    "eef_right_quat": np.s_[228:232],
    "gripper_right_qpos": np.s_[232:234],
    "gripper_right_qvel": np.s_[234:236],

    # Trunk
    "trunk_qpos": np.s_[236:240],
    "trunk_qvel": np.s_[240:244],

    # SKIP base_qpos (244-246) - NOT allowed
    # SKIP base_qpos_sin (247-249) - NOT allowed
    # SKIP base_qpos_cos (250-252) - NOT allowed

    # Base velocity (allowed)
    "base_qvel": np.s_[253:256],
}


def get_allowed_indices():
    """Get flat list of all allowed indices in order."""
    indices = []
    for key in ALLOWED_R1PRO_INDICES:
        s = ALLOWED_R1PRO_INDICES[key]
        indices.extend(range(s.start, s.stop))
    return sorted(indices)


def filter_state(state_256):
    """
    Filter 256-dim state to only allowed dimensions.

    Args:
        state_256: numpy array or torch tensor of shape (..., 256)

    Returns:
        Filtered state with only allowed dimensions
    """
    allowed_idx = get_allowed_indices()
    if hasattr(state_256, 'numpy'):  # torch tensor
        return state_256[..., allowed_idx]
    else:  # numpy array
        return state_256[..., allowed_idx]


def get_top32_indices():
    """
    Get the 32 most important state dimensions for manipulation tasks.

    Priority for "hiding Easter eggs" task:
    1. Arm joint positions (both arms) - 14 dims
    2. Gripper states (both grippers) - 4 dims
    3. End-effector positions (both arms) - 6 dims
    4. Trunk position - 4 dims
    5. Arm velocities (sample) - 4 dims
    Total: 32 dims
    """
    indices = []

    # Left arm joint positions (7 dims) - CRITICAL
    indices.extend(range(158, 165))

    # Right arm joint positions (7 dims) - CRITICAL
    indices.extend(range(197, 204))

    # Left gripper position (2 dims) - CRITICAL
    indices.extend(range(193, 195))

    # Right gripper position (2 dims) - CRITICAL
    indices.extend(range(232, 234))

    # Left end-effector position (3 dims) - IMPORTANT
    indices.extend(range(186, 189))

    # Right end-effector position (3 dims) - IMPORTANT
    indices.extend(range(225, 228))

    # Trunk position (4 dims) - IMPORTANT
    indices.extend(range(236, 240))

    # Left arm velocity (first 2 dims) - USEFUL
    indices.extend(range(179, 181))

    # Right arm velocity (first 2 dims) - USEFUL
    indices.extend(range(218, 220))

    return sorted(indices)


def filter_state_top32(state_256):
    """
    Filter 256-dim state to 32 most important dimensions.

    Args:
        state_256: numpy array or torch tensor of shape (..., 256)

    Returns:
        Filtered state with 32 most important dimensions
    """
    top32_idx = get_top32_indices()
    if hasattr(state_256, 'numpy'):  # torch tensor
        return state_256[..., top32_idx]
    else:  # numpy array
        return state_256[..., top32_idx]


def print_allowed_state_info():
    """Print information about allowed state dimensions."""
    allowed_idx = get_allowed_indices()
    print(f"Total allowed dimensions: {len(allowed_idx)}")
    print(f"\nAllowed features breakdown:")
    for key, slice_obj in ALLOWED_R1PRO_INDICES.items():
        dim_count = slice_obj.stop - slice_obj.start
        print(f"  {key:25s}: indices {slice_obj.start:3d}-{slice_obj.stop-1:3d} ({dim_count:2d} dims)")

    print(f"\nExcluded features (NOT allowed in standard track):")
    excluded = [
        ("joint_qpos[0:6]", "0-5", 6, "Base joint positions"),
        ("joint_qpos_sin[0:6]", "28-33", 6, "Sine of base joint positions"),
        ("joint_qpos_cos[0:6]", "56-61", 6, "Cosine of base joint positions"),
        ("robot_pos", "140-142", 3, "Global robot position"),
        ("robot_ori_cos", "143-145", 3, "Global orientation cosine"),
        ("robot_ori_sin", "146-148", 3, "Global orientation sine"),
        ("robot_2d_ori", "149", 1, "2D orientation"),
        ("robot_2d_ori_cos", "150", 1, "2D orientation cosine"),
        ("robot_2d_ori_sin", "151", 1, "2D orientation sine"),
        ("base_qpos", "244-246", 3, "Base joint positions"),
        ("base_qpos_sin", "247-249", 3, "Base joint sin"),
        ("base_qpos_cos", "250-252", 3, "Base joint cos"),
    ]
    total_excluded = 0
    for name, indices, count, desc in excluded:
        print(f"  {name:25s}: indices {indices:7s} ({count:2d} dims) - {desc}")
        total_excluded += count

    print(f"\nTotal excluded: {total_excluded} dimensions")
    print(f"Total allowed: {len(allowed_idx)} dimensions")
    print(f"256 - {total_excluded} = {len(allowed_idx)}")


if __name__ == "__main__":
    print_allowed_state_info()

    # Test with dummy data
    print("\n" + "="*70)
    print("Testing filter function...")
    dummy_state = np.random.randn(10, 256)  # batch of 10 states
    filtered = filter_state(dummy_state)
    print(f"Original shape: {dummy_state.shape}")
    print(f"Filtered shape: {filtered.shape}")
