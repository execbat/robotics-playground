from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator, Callable

import numpy as np

from robotics_playground.bridges.protocol import Action, Observation, ExecutionStatus

class MockBridge:
    def __init__(self) -> None:
        self._step = 0
        self._status = "mock"
        self._connected_event = asyncio.Event()
        self._connected_event.set()
        self.sim_control_calls: list[tuple[str, float | None]] = []
        self.actions_sent: list[Action] = []
        self._obs_listeners: list[Callable[[Observation], None]] = []

    @property
    def bridge_status(self) -> str:
        return self._status

    @property
    def execution_status(self) -> ExecutionStatus | None:
        return None

    async def execute_action_chunk(self, chunk: np.ndarray, timeout: float = 30.0,) -> None:
        raise NotImplementedError("MockBridge chunk execution is not implemented yet")

    async def start(self) -> None:
        self._status = "connected"
        self._connected_event.set()

    def _make_observation(self) -> Observation:
        t = self._step * 0.1
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(320) * 255 // 320
        image[:, :, 1] = int(127 + 127 * math.sin(t))

        positions = [math.sin(t + i * math.pi / 3) for i in range(6)]
        velocities = [math.cos(t + i * math.pi / 3) for i in range(6)]

        obs = Observation(
            step=self._step,
            cameras={"wrist": image},
            joint_positions=positions,
            joint_velocities=velocities,
        )
        self._step += 1
        for cb in list(self._obs_listeners):
            cb(obs)
        return obs

    async def get_observation(self) -> Observation:
        await self._connected_event.wait()
        obs = self._make_observation()
        await asyncio.sleep(0.01)
        return obs

    async def get_latest_observation(self) -> Observation | None:
        if self._status == "disconnected":
            return None
        return self._make_observation()

    async def observation_stream(self) -> AsyncIterator[Observation]:
        while True:
            yield await self.get_observation()

    async def send_action(self, action: Action) -> None:
        self.actions_sent.append(action)

    async def send_action_chunk(self, chunk: np.ndarray) -> None:
        raise NotImplementedError("MockBridge chunk execution is not implemented yet")

    async def sim_control(self, action: str, speed: float | None = None) -> None:
        self.sim_control_calls.append((action, speed))

    def add_observation_listener(self, callback: Callable[[Observation], None]) -> None:
        self._obs_listeners.append(callback)

    def remove_observation_listener(self, callback: Callable[[Observation], None]) -> None:
        with contextlib.suppress(ValueError):
            self._obs_listeners.remove(callback)

    async def close(self) -> None:
        self._status = "disconnected"
        self._connected_event.clear()

    def simulate_disconnect(self) -> None:
        self._status = "disconnected"
        self._connected_event.clear()

    def simulate_reconnect(self) -> None:
        self._status = "connected"
        self._connected_event.set()
