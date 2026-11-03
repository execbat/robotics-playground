from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from robotics_playground.bridges.protocol import Action

if TYPE_CHECKING:
    from robotics_playground.bridges.protocol import Observation
    from robotics_playground.config import EmbodimentConfig
    from robotics_playground.presets import PolicyPreset


_JOINT_LIMIT_MARGIN = 0.05


class EmbodimentAdapter:
    def __init__(
        self,
        config: EmbodimentConfig,
        camera_mapping_override: dict[str, str] | None = None,
        session_id: str = "default",
        action_type: str = "absolute",
        preset: PolicyPreset | None = None,
    ):
        self._config = config
        self._preset = preset

        if preset is not None:
            preset_mapping = preset.observation.camera_mapping

            if (
                camera_mapping_override is not None
                and camera_mapping_override != preset_mapping
            ):
                raise ValueError("Model camera mapping conflicts with preset")

            self._camera_mapping = dict(preset_mapping)
        elif camera_mapping_override is not None:
            self._camera_mapping = camera_mapping_override
        else:
            self._camera_mapping = config.camera_mapping
        self._session_id = session_id
        self._action_type = action_type

        # Build reorder indices: URDF order → training order
        self._obs_reorder = [config.joint_names.index(n) for n in config.training_order]
        # Inverse: training order → URDF order
        self._act_reorder = [config.training_order.index(n) for n in config.joint_names]

        # Pre-compute clamping arrays in URDF order
        if config.joint_limits:
            self._lower = np.array(
                [config.joint_limits[n][0] + _JOINT_LIMIT_MARGIN for n in config.joint_names]
            )
            self._upper = np.array(
                [config.joint_limits[n][1] - _JOINT_LIMIT_MARGIN for n in config.joint_names]
            )
        else:
            self._lower = None
            self._upper = None

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_mapping.keys())

    def observation_to_openpi(self, obs: Observation, instruction: str) -> dict:
        if self._preset is not None:
            return self._observation_from_preset(obs, instruction)
        result: dict = {}

        for ros_name, openpi_key in self._camera_mapping.items():
            if ros_name in obs["cameras"]:
                result[openpi_key] = obs["cameras"][ros_name]

        positions = np.array(obs["joint_positions"], dtype=np.float64)
        reordered = positions[self._obs_reorder]
        result["observation/joint_position"] = reordered.astype(np.float32)

        gripper_val = 0.0
        if len(obs["joint_positions"]) > len(self._config.joint_names):
            gripper_val = obs["joint_positions"][len(self._config.joint_names)]
        result["observation/gripper_position"] = np.array([gripper_val], dtype=np.float32)

        result["prompt"] = instruction
        result["session_id"] = self._session_id
        return result

    def action_chunk_from_openpi(
        self,
        actions_array: np.ndarray,
        current_obs: Observation | None = None,
    ) -> list[Action]:
        n_joints = len(self._config.joint_names)
        chunk_size = actions_array.shape[0]

        if self._action_type == "velocity":
            if current_obs is None:
                raise ValueError("Velocity action mode requires current_obs")
            base_pos = np.array(current_obs["joint_positions"][:n_joints], dtype=np.float64)
        else:
            base_pos = None

        result = []
        for i in range(chunk_size):
            row = actions_array[i]

            if self._action_type == "cartesian_delta":
                # Detect action format:
                # - 8-dim: joint velocity deltas (7) + gripper (1) → integrate into positions
                # - 7-dim: Cartesian deltas [pos(3), rot(3), gripper(1)] → pass through
                if row.shape[0] == n_joints + 1:  # 8-dim: joint velocities + gripper
                    if current_obs is None:
                        raise ValueError("8-dim action format requires current_obs for integration")
                    if base_pos is None:
                        base_pos = np.array(
                            current_obs["joint_positions"][:n_joints], dtype=np.float64
                        )
                    delta_training = row[:n_joints].astype(np.float64)
                    delta_urdf = delta_training[self._act_reorder]
                    base_pos = base_pos + delta_urdf
                    pos_urdf = base_pos.copy()
                    if self._lower is not None:
                        pos_urdf = np.clip(pos_urdf, self._lower, self._upper)
                    gripper_physical = float(row[n_joints])
                    g_lo, g_hi = self._config.gripper_limits
                    gripper_physical = max(g_lo, min(g_hi, gripper_physical))
                    result.append(
                        Action(
                            joint_positions=pos_urdf.tolist(),
                            joint_velocities=[math.nan] * n_joints,
                            gripper_position=gripper_physical,
                        )
                    )
                elif row.shape[0] == n_joints:
                    # 7-dim: Cartesian deltas [pos(3), rot(3), gripper(1)]
                    result.append(
                        Action(
                            joint_positions=row.tolist(),
                            joint_velocities=[math.nan] * n_joints,
                            gripper_position=float(row[-1]),
                        )
                    )
                else:
                    raise ValueError(
                        f"Unexpected action dimension: {row.shape[0]}, "
                        f"expected {n_joints} or {n_joints + 1}"
                    )
                continue

            if base_pos is not None:
                delta_training = row[:n_joints].astype(np.float64)
                delta_urdf = delta_training[self._act_reorder]
                base_pos = base_pos + delta_urdf
                pos_urdf = base_pos.copy()
            else:
                pos_physical = row[:n_joints].astype(np.float64)
                pos_urdf = pos_physical[self._act_reorder]

            if self._lower is not None:
                pos_urdf = np.clip(pos_urdf, self._lower, self._upper)

            gripper_physical = float(row[n_joints]) if row.shape[0] > n_joints else 0.0
            g_lo, g_hi = self._config.gripper_limits
            gripper_physical = max(g_lo, min(g_hi, gripper_physical))

            result.append(
                Action(
                    joint_positions=pos_urdf.tolist(),
                    joint_velocities=[math.nan] * n_joints,
                    gripper_position=gripper_physical,
                )
            )

        return result

    def _observation_from_preset(
        self,
        obs: Observation,
        instruction: str,
    ) -> dict:
        preset = self._preset
        if preset is None:
            raise RuntimeError("Observation preset is not configured")

        if "state" not in obs:
            raise ValueError("Preset requires observation.state")

        state = np.array(obs["state"], dtype=np.float32, copy=True)
        expected_shape = (preset.observation.state_dim,)

        if state.shape != expected_shape:
            raise ValueError(
                f"State shape {state.shape}; expected {expected_shape}"
            )

        if not np.isfinite(state).all():
            raise ValueError("State contains non-finite values")

        result: dict = {}

        for camera_name, model_key in self._camera_mapping.items():
            if camera_name not in obs["cameras"]:
                raise ValueError(f"Missing required camera: {camera_name}")

            result[model_key] = obs["cameras"][camera_name]

        for model_key, bounds in preset.observation.state_mapping.items():
            result[model_key] = state[bounds.start:bounds.stop].copy()

        result["prompt"] = instruction
        result["session_id"] = self._session_id

        return result
