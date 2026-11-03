from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator
from robotics_playground.presets import PolicyPreset, load_preset


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class RerunConfig(BaseModel):
    grpc_port: int = 9876
    recording_dir: str = ""


class BridgeConfig(BaseModel):
    type: str = "mock"
    watchdog_timeout: float = 10.0
    reconnect_delay: float = 2.0
    max_reconnect_delay: float = 30.0


class ROS2Config(BaseModel):
    domain_id: int = 0
    discovery_server: str | None = None
    cameras: dict[str, str] = {}
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_commands"
    state_msg_type: str = "JointState"
    physics_decimation: int = 10
    default_joint_positions: list[float] = []
    status_topic: str | None = None
    state_dim: int | None = Field(default=None, gt=0)
    action_dim: int | None = Field(default=None, gt=0)
    max_chunk_horizon: int | None = Field(default=None, gt=0)


class EmbodimentConfig(BaseModel):
    joint_names: list[str] = []
    training_order: list[str] = []
    joint_limits: dict[str, list[float]] = {}
    gripper_joint: str = ""
    gripper_limits: list[float] = [0.0, 0.04]
    camera_mapping: dict[str, str] = {}


class ModelConfig(BaseModel):
    name: str = ""
    endpoint: str
    action_horizon: int = Field(default=4, gt=0)
    action_type: str = "absolute"
    camera_mapping: dict[str, str] | None = None
    preset_id: str | None = None

    def supports_preset(self, preset: PolicyPreset | None) -> bool:
        active_id = preset.id if preset is not None else None
        return self.preset_id == active_id


class PolicyConfig(BaseModel):
    type: str = "mock"
    default_model: str = ""
    models: dict[str, ModelConfig] = {}
    embodiment: EmbodimentConfig = EmbodimentConfig()

    @model_validator(mode="after")
    def _validate_default_model(self) -> PolicyConfig:
        if self.models and self.default_model and self.default_model not in self.models:
            raise ValueError(
                f"default_model '{self.default_model}' not found in models: "
                f"{list(self.models.keys())}"
            )
        return self


class PlaygroundConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    rerun: RerunConfig = RerunConfig()
    bridge: BridgeConfig = BridgeConfig()
    ros2: ROS2Config = ROS2Config()
    policy: PolicyConfig = PolicyConfig()
    preset: PolicyPreset | None = None


def load_config(path: str | None = None) -> PlaygroundConfig:
    if path is None:
        path = os.environ.get("PLAYGROUND_CONFIG")

    if not path:
        return PlaygroundConfig()

    config_path = Path(path)

    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    if not isinstance(data, dict):
        raise ValueError("Playground config must contain a YAML mapping")

    preset_path = data.pop("preset", None)

    if preset_path is not None:
        if not isinstance(preset_path, str):
            raise ValueError("preset must be a path to a preset YAML file")

        preset_file = Path(preset_path)
        if not preset_file.is_absolute():
            preset_file = config_path.parent / preset_file

        preset = load_preset(preset_file)

        bridge = dict(data.get("bridge") or {})
        if bridge.get("type", "ros2") != "ros2":
            raise ValueError("A policy preset requires bridge.type: ros2")
        bridge["type"] = "ros2"
        data["bridge"] = bridge

        preset_ros2 = {
            "cameras": dict(preset.ros2.cameras),
            "joint_state_topic": preset.ros2.state_topic,
            "joint_command_topic": preset.ros2.action_topic,
            "state_msg_type": preset.ros2.state_msg_type,
            "physics_decimation": preset.environment.physics_decimation,
            "status_topic": preset.ros2.status_topic,
            "state_dim": preset.observation.state_dim,
            "action_dim": preset.action.dim,
            "max_chunk_horizon": preset.action.max_chunk_horizon,
        }

        ros2 = dict(data.get("ros2") or {})

        for key, value in preset_ros2.items():
            if key in ros2 and ros2[key] != value:
                raise ValueError(
                    f"ros2.{key} conflicts with preset '{preset.id}'"
                )

        ros2.update(preset_ros2)
        data["ros2"] = ros2
        data["preset"] = preset

    return PlaygroundConfig(**data)
