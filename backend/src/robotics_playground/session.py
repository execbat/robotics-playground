from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import time
from typing import TYPE_CHECKING

import numpy as np

from robotics_playground.policy import create_policy
from robotics_playground.policy.embodiment_adapter import EmbodimentAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from robotics_playground.bridges.protocol import RobotBridge
    from robotics_playground.config import PolicyConfig
    from robotics_playground.policy.protocol import PolicyClient
    from robotics_playground.rerun_logger import RerunLogger
    from robotics_playground.presets import PolicyPreset

ACTION_INTERVAL = 0.5


def _classify_connection_error(exc: BaseException) -> str:
    cause = exc.__cause__ or exc
    if isinstance(cause, socket.gaierror):
        return "server_not_found"
    if isinstance(cause, ConnectionRefusedError):
        return "server_starting"
    if isinstance(cause, (TimeoutError, OSError)):
        return "server_not_responding"
    return "connection_failed"


class Session:
    """Condition-driven inference loop.

    The loop activates automatically when three conditions are met:
    sim is running, policy is connected, and an instruction is set.
    """

    def __init__(
        self,
        bridge: RobotBridge,
        policy_config: PolicyConfig,
        rerun_logger: RerunLogger,
        observation_timeout: float = 10.0,
        preset: PolicyPreset | None = None,
    ):
        self._bridge = bridge
        self._policy_config = policy_config
        self._logger = rerun_logger
        self._observation_timeout = observation_timeout
        self._preset = preset

        self._model_id: str = ""
        self._policy: PolicyClient | None = None
        self._adapter: EmbodimentAdapter | None = None
        self._action_horizon: int = 4

        self._policy_status: str = "disconnected"
        self._policy_error: str = ""
        self._connect_task: asyncio.Task | None = None

        self._sim_state: str = "idle"
        self._instruction: str = ""
        self._step: int = 0

        self._loop_task: asyncio.Task | None = None
        self._conditions_changed = asyncio.Event()

    @property
    def policy_status(self) -> str:
        return self._policy_status

    @property
    def policy_error(self) -> str:
        return self._policy_error

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def sim_state(self) -> str:
        return self._sim_state

    @sim_state.setter
    def sim_state(self, value: str) -> None:
        self._sim_state = value
        self._conditions_changed.set()

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def step(self) -> int:
        return self._step

    @property
    def inferring(self) -> bool:
        return (
            self._sim_state == "running"
            and self._policy_status == "connected"
            and bool(self._instruction)
        )

    def set_instruction(self, text: str) -> None:
        self._instruction = text
        self._conditions_changed.set()

    def clear_instruction(self) -> None:
        self._instruction = ""
        self._step = 0
        self._conditions_changed.set()

    async def select_model(self, model_id: str) -> None:
        if model_id == self._model_id:
            return
        await self._disconnect_policy()
        if not model_id:
            self._model_id = ""
            return
        if model_id not in self._policy_config.models:
            logger.warning("Unknown model: %s", model_id)
            return
        self._model_id = model_id
        self._start_policy_connect()

    async def _disconnect_policy(self) -> None:
        if self._connect_task is not None:
            self._connect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connect_task
            self._connect_task = None
        if self._policy is not None:
            await self._policy.close()
            self._policy = None
        self._adapter = None
        self._policy_status = "disconnected"
        self._policy_error = ""
        self._conditions_changed.set()

    def _start_policy_connect(self) -> None:
        self._connect_task = asyncio.create_task(self._connect_policy_loop())

    async def _connect_policy_loop(self) -> None:
        delay = 2.0
        max_delay = 30.0
        model_id = self._model_id
        model_config = self._policy_config.models[model_id]

        self._adapter = EmbodimentAdapter(
            self._policy_config.embodiment,
            camera_mapping_override=model_config.camera_mapping,
            action_type=model_config.action_type,
            preset=self._preset,
        )
        self._action_horizon = model_config.action_horizon

        while True:
            self._policy_status = "connecting"
            self._conditions_changed.set()
            try:
                policy = create_policy(self._policy_config.type, model_config.endpoint)
                await policy.connect()
                self._policy = policy
                self._policy_status = "connected"
                self._policy_error = ""
                self._conditions_changed.set()
                logger.info("Policy connected: %s", model_id)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._policy_error = _classify_connection_error(exc)
                logger.warning(
                    "Policy connection failed for %s (%s), retrying in %.0fs",
                    model_id,
                    self._policy_error,
                    delay,
                )
                self._policy_status = "error"
                self._conditions_changed.set()
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def start_loop(self) -> None:
        if self._loop_task is not None:
            return
        self._loop_task = asyncio.create_task(self._inference_loop())

    async def stop_loop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def shutdown(self) -> None:
        await self.stop_loop()
        await self._disconnect_policy()

    async def _inference_loop(self) -> None:
        cycle = 0
        try:
            while True:
                self._conditions_changed.clear()
                if not self.inferring:
                    await self._conditions_changed.wait()
                    continue

                assert self._adapter is not None
                assert self._policy is not None

                try:
                    obs = await asyncio.wait_for(
                        self._bridge.get_observation(),
                        timeout=self._observation_timeout,
                    )
                except TimeoutError:
                    logger.warning("Observation timeout during inference")
                    continue

                if not self.inferring:
                    continue

                cycle += 1
                self._step = cycle
                obs_step = obs["step"]

                inference_request_time = time.monotonic()
                self._logger.log_instruction(
                    self._instruction, obs_step, wallclock_time=inference_request_time
                )

                try:
                    raw_action = await self._policy.infer(
                        self._adapter.observation_to_openpi(obs, self._instruction)
                    )
                except Exception as exc:
                    logger.exception("Inference failed at cycle %d", cycle)
                    self._policy_error = _classify_connection_error(exc)
                    self._policy_status = "error"
                    self._conditions_changed.set()
                    await self._disconnect_policy()
                    if self._model_id and self._model_id in self._policy_config.models:
                        self._start_policy_connect()
                    continue

                inference_response_time = time.monotonic()
                inference_ms = (inference_response_time - inference_request_time) * 1000
                logger.info("Inference cycle %d: %.1fms", cycle, inference_ms)

                if isinstance(raw_action, np.ndarray):
                    actions_tensor = raw_action
                elif isinstance(raw_action, dict):
                    actions_tensor = raw_action.get("actions", next(iter(raw_action.values())))
                else:
                    actions_tensor = raw_action

                self._logger.log_raw_action_tensor(
                    actions_tensor, obs_step, wallclock_time=inference_response_time
                )
                self._logger.log_inference_latency(
                    inference_ms, obs_step, wallclock_time=inference_response_time
                )

                action_chunk = self._adapter.action_chunk_from_openpi(
                    actions_tensor, current_obs=obs
                )
                self._logger.log_action_trajectory(
                    action_chunk, obs_step, wallclock_time=inference_response_time
                )

                horizon = action_chunk[: self._action_horizon]
                for action in horizon:
                    if not self.inferring:
                        break
                    await self._bridge.send_action(action)
                    self._step = cycle
                    await asyncio.sleep(ACTION_INTERVAL)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inference loop crashed")
