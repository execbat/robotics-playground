from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import numpy as np

from robotics_playground.bridges.protocol import Action, ExecutionStatus, Observation, ChunkExecutionInterrupted
from robotics_playground.config import BridgeConfig

if TYPE_CHECKING:
    from robotics_playground.config import ROS2Config

from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ROS2Bridge:
    def __init__(self, config: ROS2Config, bridge_config: BridgeConfig | None = None):
        self._config = config
        bc = bridge_config or BridgeConfig()
        self._watchdog_timeout = bc.watchdog_timeout
        self._reconnect_delay = bc.reconnect_delay
        self._max_reconnect_delay = bc.max_reconnect_delay

        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._obs_queue: asyncio.Queue[Observation] = asyncio.Queue(maxsize=10)
        self._step = 0
        self._latest_cameras: dict[str, np.ndarray] = {}
        self._latest_joint_positions: list[float] = []
        self._latest_joint_velocities: list[float] = []
        self._latest_state: list[float] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._status = "disconnected"
        self._publisher = None
        self._float_array_publisher = None
        self._owns_rclpy = False
        self._reset_simulation_client = None
        self._set_simulation_state_client = None
        self._step_simulation_client = None
        self._get_simulation_state_client = None
        self._sim_state: int | None = None  # STATE_STOPPED=0, STATE_PLAYING=1, STATE_PAUSED=2
        self._last_obs_time: float = 0.0
        self._connect_time: float = 0.0
        self._watchdog_task: asyncio.Task | None = None
        self._sim_paused = False
        self._last_state_query_time: float = 0.0
        self._enqueue_interval: float = 0.033  # ~30 Hz cap on cross-thread handoff
        self._last_enqueue_time: float = 0.0
        self._obs_listeners: list[Callable[[Observation], None]] = []
        self._service_call_queue: queue.Queue = queue.Queue()

        self._execution_status: ExecutionStatus | None = None
        self._execution_status_time: float = 0.0
        self._execution_status_changed = asyncio.Event()
        self._chunk_execution_lock = asyncio.Lock()
        self._connection_generation = 0

    @property
    def bridge_status(self) -> str:
        return self._status

    @property
    def execution_status(self) -> ExecutionStatus | None:
        return self._execution_status

    @property
    def sim_state(self) -> str:
        """Return sim state as string: idle/running/paused."""
        if self._sim_state is None:
            return "idle"  # Unknown state = assume idle
        state_map = {0: "idle", 1: "running", 2: "paused"}  # STATE_STOPPED/PLAYING/PAUSED
        return state_map.get(self._sim_state, "idle")

    def _setup_node(self) -> None:
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState
        from simulation_interfaces.srv import (
            GetSimulationState,
            ResetSimulation,
            SetSimulationState,
            StepSimulation,
        )
        from std_msgs.msg import Float32MultiArray, String

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self._node = Node("robotics_playground_bridge")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        for name, topic in self._config.cameras.items():
            self._node.create_subscription(
                Image,
                topic,
                lambda msg, n=name: self._handle_image(n, msg),
                sensor_qos,
            )

        use_float_array = self._config.state_msg_type == "Float32MultiArray"
        if use_float_array:
            self._node.create_subscription(
                Float32MultiArray,
                self._config.joint_state_topic,
                self._handle_float_array_state,
                sensor_qos,
            )
        else:
            self._node.create_subscription(
                JointState,
                self._config.joint_state_topic,
                self._handle_joint_state,
                sensor_qos,
            )

        if use_float_array:
            self._float_array_publisher = self._node.create_publisher(
                Float32MultiArray,
                self._config.joint_command_topic,
                10,
            )
        else:
            self._publisher = self._node.create_publisher(
                JointState,
                self._config.joint_command_topic,
                10,
            )

        if self._config.status_topic:
            self._node.create_subscription(
                String,
                self._config.status_topic,
                self._handle_execution_status,
                1,
            )
        # Standard simulation_interfaces service clients
        self._reset_simulation_client = self._node.create_client(
            ResetSimulation, "/reset_simulation"
        )
        self._set_simulation_state_client = self._node.create_client(
            SetSimulationState, "/set_simulation_state"
        )
        self._step_simulation_client = self._node.create_client(StepSimulation, "/step_simulation")
        self._get_simulation_state_client = self._node.create_client(
            GetSimulationState, "/get_simulation_state"
        )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        self._connect_time = time.monotonic()
        self._status = "connecting"
        logger.info("ROS 2 node created, status=connecting")

    def _teardown_node(self) -> None:
        self._connection_generation += 1
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self._publisher = None
        self._float_array_publisher = None
        self._reset_simulation_client = None
        self._set_simulation_state_client = None
        self._step_simulation_client = None
        self._get_simulation_state_client = None
        self._sim_state = None
        self._latest_state = None
        self._latest_joint_positions = []
        self._latest_joint_velocities = []
        self._latest_cameras.clear()

        while not self._obs_queue.empty():
            self._obs_queue.get_nowait()
        self._execution_status = None
        self._execution_status_time = 0.0
        self._status = "disconnected"
        logger.info("ROS 2 node torn down, status=disconnected")

    async def start(self) -> None:
        import rclpy

        self._loop = asyncio.get_running_loop()

        if not rclpy.ok():
            rclpy.init(domain_id=self._config.domain_id)
            self._owns_rclpy = True

        self._setup_node()
        self._watchdog_task = asyncio.create_task(self._watchdog())

    async def _reconnect(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._teardown_node)
        await loop.run_in_executor(None, self._setup_node)

    async def _query_simulation_state(self) -> None:
        """Query GetSimulationState service to sync backend view with simulator."""
        if self._get_simulation_state_client is None or self._loop is None:
            return

        from simulation_interfaces.msg import Result
        from simulation_interfaces.srv import GetSimulationState

        try:

            def on_response(f):
                try:
                    response = f.result()
                    if response.result.result != Result.RESULT_OK:
                        logger.warning(
                            "GetSimulationState failed: %s",
                            response.result.error_message,
                        )
                        return

                    old_state = self._sim_state
                    self._sim_state = response.state.state
                    if old_state is not None and old_state != response.state.state:
                        state_names = {0: "STOPPED", 1: "PLAYING", 2: "PAUSED"}
                        logger.info(
                            "Simulation state changed: %s -> %s",
                            state_names.get(old_state, old_state),
                            state_names.get(response.state.state, response.state.state),
                        )
                except Exception as exc:
                    logger.debug("GetSimulationState query failed: %s", exc)

            def result_callback(ros_future, error):
                """Called by spin thread after call_async."""
                if error:
                    logger.debug("GetSimulationState call failed: %s", error)
                    return
                ros_future.add_done_callback(on_response)

            def make_request():
                return GetSimulationState.Request()

            # Queue the service call for the spin thread
            self._service_call_queue.put(
                (self._get_simulation_state_client, make_request, result_callback)
            )
        except Exception as exc:
            logger.debug("GetSimulationState call failed: %s", exc)

    async def _watchdog(self) -> None:
        delay = self._reconnect_delay
        while True:
            await asyncio.sleep(2.0)

            if self._status == "disconnected" or self._sim_paused:
                continue

            now = time.monotonic()

            try:
                if self._status == "connecting":
                    if now - self._connect_time > self._watchdog_timeout:
                        logger.warning(
                            "No observations received after %.1fs, reconnecting...",
                            now - self._connect_time,
                        )
                        await self._reconnect()
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, self._max_reconnect_delay)
                    continue

                elapsed = now - self._last_obs_time
                if elapsed > self._watchdog_timeout:
                    logger.warning(
                        "No observations for %.1fs (timeout=%.1fs), reconnecting...",
                        elapsed,
                        self._watchdog_timeout,
                    )
                    await self._reconnect()
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self._max_reconnect_delay)
                else:
                    delay = self._reconnect_delay

                # Poll GetSimulationState periodically to detect drift
                if (
                    self._get_simulation_state_client is not None
                    and now - self._last_state_query_time > 5.0
                ):
                    self._last_state_query_time = now
                    await self._query_simulation_state()

            except Exception:
                logger.exception("Watchdog reconnect attempt failed; will retry")

    def _spin(self):
        while (executor := self._executor) is not None:
            # Process any pending service call requests from the queue
            try:
                while True:
                    call_request = self._service_call_queue.get_nowait()
                    client, request_factory, result_callback = call_request
                    try:
                        request = request_factory()
                        logger.debug(
                            "Spin thread: call_async(%s, %s)",
                            type(client).__name__,
                            type(request).__name__,
                        )
                        ros_future = client.call_async(request)
                        result_callback(ros_future, None)
                    except Exception as exc:
                        logger.warning(
                            "Spin thread: call_async failed: %s: %s",
                            type(exc).__name__,
                            exc,
                        )
                        result_callback(None, exc)
            except queue.Empty:
                pass

            executor.spin_once(timeout_sec=0.01)
            # Yield the GIL so the asyncio event loop can run.
            # Without this, the spin thread monopolises the GIL when
            # DDS callbacks arrive continuously (realtime sim mode).
            time.sleep(0.01)

    def _handle_execution_status(self, msg):
        try:
            status = ExecutionStatus.model_validate_json(msg.data)
        except ValidationError as exc:
            logger.warning("Invalid execution status: %s", exc)
            return

        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._apply_execution_status,
                status,
                time.monotonic(),
            )

    def _apply_execution_status(self, status: ExecutionStatus, received_at: float):
        self._execution_status = status
        self._execution_status_time = received_at
        self._execution_status_changed.set()

    def _handle_image(self, camera_name: str, msg):
        logger.debug("Image received: %s (%dx%d)", camera_name, msg.width, msg.height)
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self._on_image_received(camera_name, data)

    def _handle_joint_state(self, msg):
        logger.debug("Joint state received: %d joints", len(msg.position))
        self._on_joint_state_received(list(msg.position), list(msg.velocity))

    def _handle_float_array_state(self, msg):
        state = np.asarray(msg.data, dtype=np.float32)

        if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
            logger.warning("Ignoring invalid Float32MultiArray state")
            return

        expected_dim = self._config.state_dim
        if expected_dim is not None and state.size != expected_dim:
            logger.warning(
                "Ignoring state with dimension %d; expected %d",
                state.size,
                expected_dim,
            )
            return

        self._latest_state = state.tolist()

        # Compatibility with existing consumers during the migration.
        self._on_joint_state_received(list(self._latest_state), [])

    def _enqueue_observation(self):
        if self._loop is None or not self._latest_joint_positions:
            return

        now = time.monotonic()
        self._last_obs_time = now

        if self._status == "connecting":
            self._status = "connected"
            logger.info("First observation received, status=connected")
            # Query initial simulation state
            if self._loop is not None:
                self._loop.create_task(self._query_simulation_state())

        if now - self._last_enqueue_time < self._enqueue_interval:
            return
        self._last_enqueue_time = now

        obs = Observation(
            step=self._step,
            cameras=dict(self._latest_cameras),
            joint_positions=list(self._latest_joint_positions),
            joint_velocities=list(self._latest_joint_velocities),
        )
        if self._latest_state is not None:
            obs["state"] = list(self._latest_state)
        self._step += 1
        if self._step % 100 == 0:
            logger.info("Observation step %d, listeners=%d", self._step, len(self._obs_listeners))
        for cb in list(self._obs_listeners):
            cb(obs)
        self._loop.call_soon_threadsafe(self._try_put, obs)

    def _try_put(self, obs: Observation):
        with contextlib.suppress(asyncio.QueueFull):
            self._obs_queue.put_nowait(obs)

    def _on_image_received(self, camera_name: str, image: np.ndarray):
        self._latest_cameras[camera_name] = image
        self._enqueue_observation()

    def _on_joint_state_received(self, positions: list[float], velocities: list[float]):
        self._latest_joint_positions = positions
        self._latest_joint_velocities = velocities
        self._enqueue_observation()

    async def get_observation(self) -> Observation:
        obs = await self._obs_queue.get()
        while not self._obs_queue.empty():
            obs = self._obs_queue.get_nowait()
        return obs

    async def get_latest_observation(self) -> Observation | None:
        if not self._latest_joint_positions or not self._latest_cameras:
            return None

        obs = Observation(
            step=self._step,
            cameras=dict(self._latest_cameras),
            joint_positions=list(self._latest_joint_positions),
            joint_velocities=list(self._latest_joint_velocities),
        )

        if self._latest_state is not None:
            obs["state"] = list(self._latest_state)

        return obs

    async def observation_stream(self) -> AsyncIterator[Observation]:
        while True:
            obs = await self._obs_queue.get()
            yield obs

    async def send_action(self, action: Action) -> None:
        if self._node is None:
            return
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32MultiArray

        if self._publisher is not None:
            positions = [float(p) for p in action["joint_positions"]] + [
                float(action["gripper_position"])
            ]
            velocities = [float(v) for v in action["joint_velocities"]] + [float("nan")]
            msg = JointState()
            msg.position = positions
            msg.velocity = velocities
            self._publisher.publish(msg)

        if self._float_array_publisher is not None:
            msg = Float32MultiArray()
            msg.data = [float(v) for v in action["joint_positions"]]
            self._float_array_publisher.publish(msg)

    async def send_action_chunk(self, chunk: np.ndarray) -> None:
        """Publish a complete chunk; the environment controls execution timing."""
        if self._node is None or self._float_array_publisher is None:
            raise RuntimeError("Chunk transport requires a connected Float32MultiArray publisher")

        action_dim = self._config.action_dim
        if action_dim is None:
            raise ValueError("ros2.action_dim must be configured for chunk transport")

        values = np.array(chunk, dtype=np.float32, copy=True)

        if values.ndim != 2 or values.shape[1] != action_dim:
            raise ValueError(f"Expected action chunk [H,{action_dim}], got {values.shape}")

        if not np.isfinite(values).all():
            raise ValueError("Action chunk must contain only finite values")

        horizon = values.shape[0]
        max_horizon = self._config.max_chunk_horizon

        if max_horizon is not None and horizon > max_horizon:
            raise ValueError(f"Chunk horizon {horizon} exceeds configured maximum {max_horizon}")

        if self._float_array_publisher.get_subscription_count() == 0:
            raise RuntimeError("No environment subscriber on the action topic")

        from std_msgs.msg import Float32MultiArray, MultiArrayDimension

        msg = Float32MultiArray()
        msg.layout.data_offset = 0

        if horizon > 0:
            msg.layout.dim = [
                MultiArrayDimension(label="horizon", size=horizon, stride=horizon * action_dim,),
                MultiArrayDimension(label="action",  size=action_dim, stride=action_dim,),
            ]

        msg.data = values.reshape(-1).tolist()
        self._float_array_publisher.publish(msg)

    async def execute_action_chunk(self, chunk: np.ndarray, timeout: float = 30.0,) -> None:
        """Send one chunk and wait for acceptance and completion."""
        if not 0 < timeout < float("inf"):
            raise ValueError("timeout must be positive and finite")

        async with self._chunk_execution_lock:
            status = self.execution_status

            if status is None:
                raise RuntimeError("No execution status received from environment")

            if time.monotonic() - self._execution_status_time > self._watchdog_timeout:
                raise RuntimeError("Environment execution status is stale")

            if status.paused:
                raise ChunkExecutionInterrupted("Environment is paused")

            if not status.idle or status.remaining != 0:
                raise RuntimeError("Environment still has an unfinished chunk")

            previous_count = status.received_chunks
            previous_sim_time = status.sim_time
            generation = self._connection_generation
            sent_at = time.monotonic()

            await self.send_action_chunk(chunk)

            deadline = time.monotonic() + timeout

            while True:
                self._execution_status_changed.clear()

                if generation != self._connection_generation:
                    raise ChunkExecutionInterrupted("ROS connection changed during execution")

                status = self.execution_status
                now = time.monotonic()

                if status is None:
                    raise ChunkExecutionInterrupted("Environment execution status was cleared")

                if now - self._execution_status_time > self._watchdog_timeout:
                    raise TimeoutError("Environment stopped publishing execution status")

                if self._execution_status_time > sent_at:
                    if status.paused:
                        raise ChunkExecutionInterrupted("Environment paused or stopped during execution")

                    if status.sim_time < previous_sim_time:
                        raise ChunkExecutionInterrupted("Environment simulation time was reset")

                    if status.received_chunks < previous_count:
                        raise ChunkExecutionInterrupted("Environment chunk counter was reset")

                    if status.received_chunks > previous_count + 1:
                        raise ChunkExecutionInterrupted("Another action message replaced the submitted chunk")

                    accepted = status.received_chunks == previous_count + 1

                    if accepted and status.remaining == 0 and status.idle:
                        return

                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for chunk acceptance/completion")

                try:
                    await asyncio.wait_for(
                        self._execution_status_changed.wait(),
                        timeout=min(remaining, 0.5),
                    )
                except TimeoutError:
                    pass

    async def _call_service_async(self, client, request_factory, timeout: float = 5.0):
        """Thread-safe service call that bridges ROS2 to asyncio.

        Queues the service call to be executed by the spin thread (the ONLY thread
        where call_async is safe), then bridges the ROS2 future to asyncio.

        Args:
            client: ROS2 service client
            request_factory: Callable that creates Request() in spin thread
            timeout: Service call timeout in seconds
        """
        loop = asyncio.get_running_loop()
        asyncio_future = loop.create_future()
        call_queued = threading.Event()

        def result_callback(ros_future, error):
            """Called by spin thread after call_async completes or errors."""
            call_queued.set()
            if error:
                loop.call_soon_threadsafe(asyncio_future.set_exception, error)
                return

            def on_done(f):
                """Called by rclpy executor when service completes."""
                if asyncio_future.done():
                    return  # Already timed out or cancelled
                try:
                    result = f.result()
                    loop.call_soon_threadsafe(asyncio_future.set_result, result)
                except Exception as exc:
                    loop.call_soon_threadsafe(asyncio_future.set_exception, exc)

            ros_future.add_done_callback(on_done)

        # Queue the service call for the spin thread to execute
        self._service_call_queue.put((client, request_factory, result_callback))

        # Wait for the spin thread to pick up and execute the call
        await asyncio.get_event_loop().run_in_executor(None, call_queued.wait, 1.0)

        if not call_queued.is_set():
            raise RuntimeError("Service call was not picked up by spin thread")

        try:
            return await asyncio.wait_for(asyncio_future, timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"Service call timed out after {timeout}s") from exc

    async def sim_control(self, action: str, speed: float | None = None) -> None:
        if self._node is None:
            return

        from simulation_interfaces.msg import Result
        from simulation_interfaces.srv import (
            ResetSimulation,
            SetSimulationState,
            StepSimulation,
        )

        if action in ("play", "pause", "stop"):
            state_map = {"stop": 0, "play": 1, "pause": 2}
            if self._set_simulation_state_client is not None:
                target_state = state_map[action]

                def make_request():
                    req = SetSimulationState.Request()
                    req.state.state = target_state
                    return req

                try:
                    response = await self._call_service_async(
                        self._set_simulation_state_client, make_request, timeout=5.0
                    )
                    if response.result.result != Result.RESULT_OK:
                        err = response.result.error_message
                        logger.warning("SetSimulationState(%s) failed: %s", action, err)
                        raise RuntimeError(f"SetSimulationState({action}) failed: {err}")
                    was_paused = self._sim_paused
                    self._sim_paused = action in ("pause", "stop")
                    if was_paused and not self._sim_paused:
                        self._last_obs_time = time.monotonic()
                    logger.info("SetSimulationState(%s) completed successfully", action)
                except TimeoutError:
                    logger.warning("SetSimulationState(%s) timed out", action)
                    raise
                except Exception as exc:
                    logger.warning("SetSimulationState(%s) failed: %s", action, exc)
                    raise

        elif action == "step":
            if self._step_simulation_client is not None:
                num_steps = self._config.physics_decimation

                def make_request():
                    req = StepSimulation.Request()
                    req.steps = num_steps
                    return req

                try:
                    response = await self._call_service_async(
                        self._step_simulation_client, make_request, timeout=5.0
                    )
                    if response.result.result != Result.RESULT_OK:
                        err = response.result.error_message
                        logger.warning("StepSimulation failed: %s", err)
                        raise RuntimeError(f"StepSimulation failed: {err}")
                    logger.debug("StepSimulation(%d) completed successfully", num_steps)
                except TimeoutError:
                    logger.warning("StepSimulation timed out")
                    raise
                except Exception as exc:
                    logger.warning("StepSimulation failed: %s", exc)
                    raise

        elif action == "reset":
            if self._reset_simulation_client is not None:

                def make_request():
                    return ResetSimulation.Request()

                try:
                    response = await self._call_service_async(
                        self._reset_simulation_client, make_request, timeout=5.0
                    )
                    if response.result.result != Result.RESULT_OK:
                        err = response.result.error_message
                        logger.warning("ResetSimulation failed: %s", err)
                        raise RuntimeError(f"ResetSimulation failed: {err}")
                    # Reset transitions simulator to STOPPED
                    self._step = 0
                    self._sim_paused = True
                    logger.info("ResetSimulation completed successfully")
                except TimeoutError:
                    logger.warning("ResetSimulation timed out")
                    raise
                except Exception as exc:
                    logger.warning("ResetSimulation failed: %s", exc)
                    raise

    def add_observation_listener(self, callback: Callable[[Observation], None]) -> None:
        self._obs_listeners.append(callback)

    def remove_observation_listener(self, callback: Callable[[Observation], None]) -> None:
        with contextlib.suppress(ValueError):
            self._obs_listeners.remove(callback)

    async def close(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None
        self._teardown_node()
        if self._owns_rclpy:
            import rclpy

            rclpy.shutdown()
            self._owns_rclpy = False
