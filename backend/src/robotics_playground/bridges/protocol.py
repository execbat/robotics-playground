from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from collections.abc import AsyncIterator, Callable
from typing import Protocol, TypedDict

import numpy as np

class ChunkExecutionInterrupted(RuntimeError):
    """Execution interrupted by pause, reset or connection change."""

class ExecutionStatus(BaseModel):
    """Execution feedback shared by all ROS environments."""

    model_config = ConfigDict(frozen=True, strict=True)

    sim_time: float = Field(ge=0, allow_inf_nan=False)
    control_hz: float = Field(gt=0, allow_inf_nan=False)
    remaining: int = Field(ge=0)
    idle: bool
    paused: bool
    received_chunks: int = Field(ge=0)


class Observation(TypedDict):
    step: int
    cameras: dict[str, np.ndarray]
    joint_positions: list[float]
    joint_velocities: list[float]


class Action(TypedDict):
    joint_positions: list[float]
    joint_velocities: list[float]
    gripper_position: float


class RobotBridge(Protocol):
    @property
    def bridge_status(self) -> str: ...

    @property
    def execution_status(self) -> ExecutionStatus | None: ...

    async def start(self) -> None: ...

    async def get_observation(self) -> Observation: ...

    async def get_latest_observation(self) -> Observation | None: ...

    def observation_stream(self) -> AsyncIterator[Observation]: ...

    async def send_action(self, action: Action) -> None: ...

    async def send_action_chunk(self, chunk: np.ndarray) -> None: ...

    async def sim_control(self, action: str, speed: float | None = None) -> None: ...

    def add_observation_listener(self, callback: Callable[[Observation], None]) -> None: ...

    def remove_observation_listener(self, callback: Callable[[Observation], None]) -> None: ...

    async def close(self) -> None: ...

    async def execute_action_chunk(self, chunk: np.ndarray, timeout: float = 30.0,) -> None: ...
