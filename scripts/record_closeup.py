"""Record close-up gear assembly videos with different camera angles.

Usage:
    ./isaaclab.sh -p scripts/record_closeup.py --angle A --checkpoint /path/to/model_9.pt
    ./isaaclab.sh -p scripts/record_closeup.py --angle all --checkpoint /path/to/model_9.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

CAMERA_OPTIONS = {
    "A": {
        "eye": (-0.65, 0.55, 0.15),
        "lookat": (-1.02, 0.21, -0.05),
        "desc": "Front-right, slightly above — classic engineering view",
    },
    "B": {
        "eye": (-1.0, 0.55, 0.05),
        "lookat": (-1.02, 0.21, -0.05),
        "desc": "Side view, eye-level with gripper",
    },
    "C": {
        "eye": (-0.80, 0.35, 0.30),
        "lookat": (-1.02, 0.21, -0.08),
        "desc": "Top-down angled, insertion focus",
    },
    "D": {
        "eye": (-0.72, 0.20, -0.05),
        "lookat": (-1.02, 0.21, 0.0),
        "desc": "Front, looking up at fingers",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Record close-up gear assembly video")
    parser.add_argument("--angle", type=str, default="A", help="Camera angle: A, B, C, D, or all")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--video_length", type=int, default=400)
    parser.add_argument("--output_dir", type=str, default=None)
    args, unknown = parser.parse_known_args()

    # -- bootstrap Isaac Sim ------------------------------------------------
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    # -- heavy imports after app launch -------------------------------------
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401 – registers Gymnasium envs

    from rsl_rl.runners import OnPolicyRunner

    angle_key = args.angle.upper()
    if angle_key not in CAMERA_OPTIONS:
        print(f"Unknown angle '{angle_key}'. Choose from: {list(CAMERA_OPTIONS.keys())}")
        simulation_app.close()
        return

    cam = CAMERA_OPTIONS[angle_key]
    print(f"\n{'='*60}")
    print(f"Angle {angle_key}: {cam['desc']}")
    print(f"  Eye   : {cam['eye']}")
    print(f"  Lookat: {cam['lookat']}")
    print(f"{'='*60}\n")

    # Determine output dir
    ckpt_dir = os.path.dirname(args.checkpoint)
    out_dir = args.output_dir or os.path.join(ckpt_dir, "videos", "closeup")
    os.makedirs(out_dir, exist_ok=True)

    # Build the env with 1 robot, video mode on
    task_name = "Isaac-Deploy-GearAssembly-UR10e-2F140-v0"
    env = gym.make(task_name, num_envs=1, render_mode="rgb_array")

    # Patch camera position for close-up
    unwrapped = env.unwrapped
    unwrapped.cfg.viewer.eye = cam["eye"]
    unwrapped.cfg.viewer.lookat = cam["lookat"]
    # Also patch the video recorder backend if already created
    if hasattr(unwrapped, "video_recorder") and unwrapped.video_recorder is not None:
        vr = unwrapped.video_recorder
        if hasattr(vr, "_backend") and vr._backend is not None:
            vr._backend.cfg.camera_position = cam["eye"]
            vr._backend.cfg.camera_target = cam["lookat"]
        if hasattr(vr, "cfg"):
            vr.cfg.camera_position = cam["eye"]
            vr.cfg.camera_target = cam["lookat"]

    # Load the RSL-RL agent config
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

    agent_cfg: RslRlOnPolicyRunnerCfg = parse_env_cfg(
        task_name,
        device="cuda:0",
        num_envs=1,
        use_fabric=True,
    )[1]
    agent_cfg.resume = True

    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    # Reset and run
    obs, info = env.reset()
    print(f"Running policy for {args.video_length} steps...")
    for step_i in range(args.video_length):
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, terminated, truncated, info = env.step(actions)

    # The gym RecordVideo wrapper should have saved the mp4
    # Find it
    import glob
    vids = glob.glob(os.path.join(ckpt_dir, "videos", "**", "*.mp4"), recursive=True)
    vids.sort(key=os.path.getmtime, reverse=True)

    if vids:
        latest = vids[0]
        dest = os.path.join(out_dir, f"gear_assembly_angle_{angle_key}.mp4")
        import shutil
        shutil.copy2(latest, dest)
        print(f"\n✅ Video saved: {dest}")
    else:
        print("\n⚠️ No video file found — check gym RecordVideo wrapper")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
