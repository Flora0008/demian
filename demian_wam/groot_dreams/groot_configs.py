from groot_dreams.data.dataset import ModalityConfig
from groot_dreams.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoGridStitch, VideoColorJitter
from groot_dreams.data.transform.base import ComposedModalityTransform
from groot_dreams.data.transform.concat import ConcatTransform
from groot_dreams.data.transform.state_action import StateActionToTensor, StateActionTransform


def construct_modality_config_and_transforms(
    num_frames,
    embodiment,
    agibot_pad_freq10=False,
    waist_concat=False,
    use_image_aug=False,
    video_height=None,
    video_width=None,
    robocasa_video_mode="grid",
):
    if embodiment == "gr1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view_freq20"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
        }
    elif embodiment == "g1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_leg",
                    "state.right_leg",
                    "state.waist",
                    "state.left_arm",
                    "state.left_hand",
                    "state.right_arm",
                    "state.right_hand",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_leg",
                    "action.right_leg",
                    "action.waist",
                    "action.left_arm",
                    "action.left_hand",
                    "action.right_arm",
                    "action.right_hand",
                ],
            ),
        }
    elif embodiment == "yam":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.top_camera-images-rgb"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.ee_pose_obs_left",
                    "state.ee_pose_obs_right",
                    "state.gripper_pos_obs_left",
                    "state.gripper_pos_obs_right",
                    "state.joint_pos_obs_left",
                    "state.joint_pos_obs_right",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.ee_pose_action_left",
                    "action.ee_pose_action_right",
                    "action.gripper_pos_action_left",
                    "action.gripper_pos_action_right",
                    "action.joint_pos_action_left",
                    "action.joint_pos_action_right",
                ],
            ),
        }
    elif embodiment == "agibot":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.top_head" if not agibot_pad_freq10 else "video.top_head_pad_freq10"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_position",
                ] if waist_concat else [
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_pitch",
                    "state.waist_lift",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_position",
                    "action.robot_velocity",
                ] if waist_concat else [
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_pitch",
                    "action.waist_lift",
                    "action.robot_velocity",
                ],
            ),
        }
    elif embodiment == "robocasa":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_keys = [
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ]
        grid_output_key = "video.grid"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.base_position",
                    "state.base_rotation",
                    "state.end_effector_position_relative",
                    "state.end_effector_rotation_relative",
                    "state.gripper_qpos",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.end_effector_position",
                    "action.end_effector_rotation",
                    "action.gripper_close",
                ],
            ),
        }

    video_modality, state_modality, action_modality = config["video"], config["state"], config["action"]
    height = video_height if video_height is not None else 480
    width = video_width if video_width is not None else 640

    # For robocasa, the default training input is the 3-view 2x2 grid.
    # Ablations can select a single camera as the diffusion video while leaving
    # the other transformed views in the sample for Reason1 conditioning.
    if embodiment == "robocasa" and robocasa_video_mode == "grid":
        video_transform_keys = [grid_output_key]
        video_concat_order = [grid_output_key]
    elif embodiment == "robocasa" and robocasa_video_mode == "left_reason1_right_wrist":
        video_transform_keys = video_modality.modality_keys
        video_concat_order = ["video.robot0_agentview_left"]
    elif embodiment == "robocasa":
        raise ValueError(f"Unsupported robocasa_video_mode: {robocasa_video_mode}")
    else:
        video_transform_keys = video_modality.modality_keys
        video_concat_order = video_modality.modality_keys

    def _make_video_head_transforms():
        """Create fresh video transform instances (one set per train/test)."""
        if embodiment == "robocasa" and robocasa_video_mode == "grid":
            return [
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoGridStitch(
                    apply_to=video_modality.modality_keys,
                    input_keys=video_modality.modality_keys,
                    output_key=grid_output_key,
                ),
            ]
        return [VideoToTensor(apply_to=video_modality.modality_keys)]

    # Image augmentation (train only) — matches cosmos-policy's augmentation
    _aug_transforms = []
    if use_image_aug:
        _aug_transforms.append(
            VideoColorJitter(
                apply_to=video_transform_keys,
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05,
            )
        )

    train_transform = ComposedModalityTransform(
        transforms=[
            *_make_video_head_transforms(),
            VideoCrop(apply_to=video_transform_keys, target_width=width, target_height=height),
            VideoResize(apply_to=video_transform_keys, height=height, width=width, interpolation="linear"),
            *_aug_transforms,

            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
                key: "min_max" for key in state_modality.modality_keys
            }),

            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
                key: "min_max" for key in action_modality.modality_keys
            }),

            ConcatTransform(
                video_concat_order=video_concat_order,
                state_concat_order=state_modality.modality_keys,
                action_concat_order=action_modality.modality_keys,
            ),
        ]
    )
    test_transform = ComposedModalityTransform(
        transforms=[
            *_make_video_head_transforms(),
            VideoCrop(apply_to=video_transform_keys, target_width=width, target_height=height),
            VideoResize(apply_to=video_transform_keys, height=height, width=width, interpolation="linear"),

            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
                key: "min_max" for key in state_modality.modality_keys
            }),

            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
                key: "min_max" for key in action_modality.modality_keys
            }),

            ConcatTransform(
                video_concat_order=video_concat_order,
                state_concat_order=state_modality.modality_keys,
                action_concat_order=action_modality.modality_keys,
            ),
        ]
    )

    return config, train_transform, test_transform
