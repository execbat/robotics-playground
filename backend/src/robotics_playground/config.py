from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


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


def load_config(path: str | None = None) -> PlaygroundConfig:
    if path is None:
        path = os.environ.get("PLAYGROUND_CONFIG")
    if path and Path(path).is_file():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return PlaygroundConfig(**data)
    return PlaygroundConfig()
