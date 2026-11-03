from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PresetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentPreset(PresetSection):
    id: str
    module: str
    physics_hz: int = Field(gt=0)
    control_hz: int = Field(gt=0)
    physics_decimation: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_frequency(self) -> EnvironmentPreset:
        if self.control_hz * self.physics_decimation != self.physics_hz:
            raise ValueError(
                "control_hz * physics_decimation must equal physics_hz"
            )
        return self


class ROS2Preset(PresetSection):
    cameras: dict[str, str]
    state_topic: str
    state_msg_type: Literal["Float32MultiArray"]
    action_topic: str
    status_topic: str


class StateSlice(PresetSection):
    start: int = Field(ge=0)
    stop: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> StateSlice:
        if self.stop <= self.start:
            raise ValueError("State slice stop must be greater than start")
        return self


class ObservationPreset(PresetSection):
    state_dim: int = Field(gt=0)
    state_components: list[str]
    state_mapping: dict[str, StateSlice]
    camera_mapping: dict[str, str]

    @model_validator(mode="after")
    def validate_state(self) -> ObservationPreset:
        if len(self.state_components) != self.state_dim:
            raise ValueError("state_components length must equal state_dim")

        if not self.state_mapping:
            raise ValueError("state_mapping must not be empty")

        for key, bounds in self.state_mapping.items():
            if bounds.stop > self.state_dim:
                raise ValueError(f"State slice exceeds state_dim: {key}")

        return self


class ActionPreset(PresetSection):
    dim: int = Field(gt=0)
    motion_type: Literal["joint_velocity", "cartesian_delta"]
    gripper_type: Literal["closure", "binary_command"]
    components: list[str]
    max_chunk_horizon: int = Field(gt=0)
    execute_horizon: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_action(self) -> ActionPreset:
        if len(self.components) != self.dim:
            raise ValueError("components length must equal action dim")

        if self.execute_horizon > self.max_chunk_horizon:
            raise ValueError(
                "execute_horizon must not exceed max_chunk_horizon"
            )

        return self


class ProcessingPreset(PresetSection):
    observation_normalization: Literal["server"]
    action_denormalization: Literal["server"]
    action_delivery: Literal["chunk"]


class PolicyPreset(PresetSection):
    id: str
    checkpoint: str
    environment: EnvironmentPreset
    ros2: ROS2Preset
    observation: ObservationPreset
    action: ActionPreset
    processing: ProcessingPreset

    @model_validator(mode="after")
    def validate_cameras(self) -> PolicyPreset:
        sources = set(self.ros2.cameras)
        mapped = set(self.observation.camera_mapping)

        if sources != mapped:
            raise ValueError(
                "ROS camera names must match observation camera mapping"
            )

        return self


def load_preset(path: str | Path) -> PolicyPreset:
    path = Path(path)

    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError(f"Preset must contain a YAML mapping: {path}")

    preset = PolicyPreset.model_validate(data)

    if preset.id != path.stem:
        raise ValueError(
            f"Preset id '{preset.id}' must match filename '{path.stem}'"
        )

    return preset
