# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.deploy.mdp.events as gear_assembly_events
from isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.gear_assembly_env_cfg import GearAssemblyEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


##
# Gripper-specific helper functions
##


def set_finger_joint_pos_panda(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Franka Panda hand.

    The Panda hand has 2 prismatic finger joints (panda_finger_joint1, panda_finger_joint2).
    Both joints are set to the same target position (symmetric parallel gripper).

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of finger joint indices (2 joints expected)
        finger_joint_position: Target position for finger joints (in meters, 0.0 to 0.04)
    """
    for idx in reset_ind_joint_pos:
        if len(finger_joints) < 2:
            raise ValueError(f"Panda hand requires at least 2 finger joints, got {len(finger_joints)}")

        # Both finger joints move symmetrically
        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "friction_distribution_params": (0.3, 0.7),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    small_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    medium_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_medium", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    large_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_large", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    gear_base_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_base", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_.*finger"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_gear_type = EventTerm(
        func=gear_assembly_events.randomize_gear_type,
        mode="reset",
        params={"gear_types": ["gear_small", "gear_medium", "gear_large"]},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_gears_and_base_pose = EventTerm(
        func=gear_assembly_events.randomize_gears_and_base_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.10, 0.10],
                "z": [-0.05, 0.05],
                "roll": [-math.pi / 90, math.pi / 90],  # 2 degree
                "pitch": [-math.pi / 90, math.pi / 90],  # 2 degree
                "yaw": [-math.pi / 6, math.pi / 6],  # 30 degree
            },
            "gear_pos_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0575, 0.0775],
            },
            "velocity_range": {},
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=gear_assembly_events.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.005, 0.005], "z": [-0.003, 0.003]},
        },
    )


@configclass
class FrankaGearAssemblyEnvCfg(GearAssemblyEnvCfg):
    """Configuration for Franka Panda Gear Assembly Environment.

    The Franka Emika Panda is a 7-DOF collaborative robot arm equipped with the
    Panda parallel jaw gripper for gear manipulation tasks.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Robot-specific parameters for Franka Panda
        self.end_effector_body_name = "panda_hand"  # End effector body name for IK
        self.num_arm_joints = 7  # Number of arm joints (Franka Panda has 7 DOF)
        # Rotation offset for grasp pose (quaternion [x, y, z, w])
        # Franka's panda_hand frame: z-axis points forward (along fingers), x-axis points down
        # For downward grasp: rotate to align gripper fingers with gear (same as Rizon 4s)
        self.grasp_rot_offset = [
            -0.707,
            0.707,
            0.0,
            0.0,
        ]
        self.gripper_joint_setter_func = set_finger_joint_pos_panda  # Panda hand joint setter function

        # Gear orientation termination thresholds (in degrees)
        self.gear_orientation_roll_threshold_deg = 15.0  # Maximum allowed roll deviation
        self.gear_orientation_pitch_threshold_deg = 15.0  # Maximum allowed pitch deviation
        self.gear_orientation_yaw_threshold_deg = 180.0  # Maximum allowed yaw deviation

        # Observation configuration for Franka Panda joints (arm only, not gripper)
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        # override events
        self.events = EventCfg()

        # Update termination thresholds from config
        self.terminations.gear_orientation_exceeded.params["roll_threshold_deg"] = (
            self.gear_orientation_roll_threshold_deg
        )
        self.terminations.gear_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.gear_orientation_pitch_threshold_deg
        )
        self.terminations.gear_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.gear_orientation_yaw_threshold_deg
        )

        # Action configuration for Franka Panda arm (7 DOF)
        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )

        # ── Scene geometry adjustments ──────────────────────────────────────
        # The base scene places gears at (-1.02, 0.21) — designed for UR10e (1.3m reach).
        # Franka Panda reach is only 0.855m, so we move the gear assembly closer.
        # Move gear base and all gears from (-1.02, 0.21) to (-0.50, 0.15)
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        gear_pos = (-0.50, 0.15, -0.1)
        gear_rot = (0.0, 0.0, 0.70711, 0.70711)
        self.scene.factory_gear_base.init_state.pos = gear_pos
        self.scene.factory_gear_base.init_state.rot = gear_rot
        self.scene.factory_gear_small.init_state.pos = gear_pos
        self.scene.factory_gear_small.init_state.rot = gear_rot
        self.scene.factory_gear_medium.init_state.pos = gear_pos
        self.scene.factory_gear_medium.init_state.rot = gear_rot
        self.scene.factory_gear_large.init_state.pos = gear_pos
        self.scene.factory_gear_large.init_state.rot = gear_rot

        # Switch robot to Franka Panda with high PD gains
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            # Joint positions: arm extended toward the gear workspace
            # IK will refine this during set_robot_to_grasp_pose, but start close
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.310,
                    "panda_joint5": 0.0,
                    "panda_joint6": 2.0,
                    "panda_joint7": 0.741,
                    "panda_finger_joint.*": 0.04,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # Panda hand actuator configuration for gear manipulation
        # Override from HIGH_PD_CFG to use appropriate values for grasping gears
        self.scene.robot.actuators["panda_hand"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        )

        # Gear offsets and grasp positions for Franka Panda hand
        # Z offset: distance from panda_hand frame to gear center when grasped
        # Panda fingertip is ~0.1034m below panda_hand origin;
        # gear should sit lower in the fingers (not at the base), so use -0.13
        self.gear_offsets_grasp = {
            "gear_small": [0.0, -self.gear_offsets["gear_small"][0], -0.13],
            "gear_medium": [0.0, -self.gear_offsets["gear_medium"][0], -0.135],  # 0.5cm lower than previous -0.126
            "gear_large": [0.0, -self.gear_offsets["gear_large"][0], -0.13],
        }

        # Grasp widths for Panda hand (in meters, per finger)
        # Panda fingers: 0.0 = fully closed, 0.04 = fully open (per finger)
        # grasp_width = initial finger opening when gear is placed (just wide enough to fit)
        # close_width = PD target to grip the gear tight (controller drives fingers to this)
        # Small gear is tiny — must start nearly closed or gear slips through
        self.hand_grasp_width = {
            "gear_small": 0.010,   # barely open — gear must not fall through
            "gear_medium": 0.020,
            "gear_large": 0.025,
        }

        # Close widths: PD target for gripping (tighter than grasp_width)
        self.hand_close_width = {
            "gear_small": 0.005,   # near-closed for tiny gear
            "gear_medium": 0.015,
            "gear_large": 0.020,
        }

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        # Populate termination term parameters
        self.terminations.gear_dropped.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.terminations.gear_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class FrankaGearAssemblyEnvCfg_PLAY(FrankaGearAssemblyEnvCfg):
    """Play configuration for Franka Panda gear assembly."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
