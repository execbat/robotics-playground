from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class FakeResult:
    RESULT_OK = 1
    RESULT_OPERATION_FAILED = 4


class FakeRos2Future:
    """Simulates a ROS2 future whose callback is triggered by the rclpy spin thread."""

    def __init__(self):
        self._callbacks = []
        self._result = None
        self._exception = None
        self._done = False

    def add_done_callback(self, cb):
        self._callbacks.append(cb)
        if self._done:
            cb(self)

    def done(self):
        return self._done

    def result(self):
        if self._exception:
            raise self._exception
        return self._result

    def set_result(self, result):
        self._result = result
        self._done = True
        for cb in self._callbacks:
            cb(self)

    def set_exception(self, exc):
        self._exception = exc
        self._done = True
        for cb in self._callbacks:
            cb(self)


@pytest.fixture
def mock_rclpy():
    mock_node = MagicMock()
    mock_node.create_subscription = MagicMock()
    mock_node.create_publisher = MagicMock()
    mock_node.create_client = MagicMock()
    mock_node.destroy_node = MagicMock()

    mock_rclpy_module = MagicMock()
    mock_rclpy_module.init = MagicMock()
    mock_rclpy_module.ok = MagicMock(return_value=False)
    mock_rclpy_module.shutdown = MagicMock()

    mock_node_class = MagicMock(return_value=mock_node)

    mocks = {
        "rclpy": mock_rclpy_module,
        "rclpy.node": MagicMock(Node=mock_node_class),
        "rclpy.executors": MagicMock(),
        "rclpy.qos": MagicMock(),
        "sensor_msgs": MagicMock(),
        "sensor_msgs.msg": MagicMock(),
        "std_msgs": MagicMock(),
        "std_msgs.msg": MagicMock(),
        "simulation_interfaces": MagicMock(),
        "simulation_interfaces.srv": MagicMock(),
        "simulation_interfaces.msg": MagicMock(Result=FakeResult),
    }

    with patch.dict("sys.modules", mocks):
        yield {
            "rclpy": mock_rclpy_module,
            "node": mock_node,
            "node_class": mock_node_class,
        }


def test_ros2_bridge_initial_status_is_disconnected(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    assert bridge.bridge_status == "disconnected"


@pytest.mark.anyio
async def test_ros2_bridge_start_initializes_rclpy(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    mock_rclpy["rclpy"].init.assert_called_once()
    assert bridge.bridge_status == "connecting"
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_close_shuts_down(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    await bridge.close()
    mock_rclpy["rclpy"].shutdown.assert_called_once()
    assert bridge.bridge_status == "disconnected"


@pytest.mark.anyio
async def test_ros2_bridge_close_skips_shutdown_when_not_owner(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    mock_rclpy["rclpy"].ok = MagicMock(return_value=True)
    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    mock_rclpy["rclpy"].init.assert_not_called()
    await bridge.close()
    mock_rclpy["rclpy"].shutdown.assert_not_called()
    assert bridge.bridge_status == "disconnected"


@pytest.mark.anyio
async def test_ros2_bridge_observation_from_callbacks(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    config = ROS2Config(cameras={"wrist": "/cam/wrist"})
    bridge = ROS2Bridge(config)
    await bridge.start()
    bridge._enqueue_interval = 0

    bridge._on_joint_state_received([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
    image_data = np.zeros((240, 320, 3), dtype=np.uint8)
    bridge._on_image_received("wrist", image_data)

    obs = None
    async for o in bridge.observation_stream():
        obs = o
        if "wrist" in obs.get("cameras", {}):
            break

    assert obs is not None
    assert "wrist" in obs["cameras"]
    assert obs["joint_positions"] == [0.1, 0.2, 0.3]
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_get_observation(mock_rclpy):
    from robotics_playground.bridges.protocol import Observation
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    config = ROS2Config(cameras={"wrist": "/cam/wrist"})
    bridge = ROS2Bridge(config)
    await bridge.start()

    # Directly enqueue a test observation
    image_data = np.zeros((240, 320, 3), dtype=np.uint8)
    test_obs = Observation(
        step=0,
        cameras={"wrist": image_data},
        joint_positions=[0.1, 0.2, 0.3],
        joint_velocities=[0.01, 0.02, 0.03],
    )
    await bridge._obs_queue.put(test_obs)

    obs = await bridge.get_observation()

    assert obs is not None
    assert "wrist" in obs["cameras"]
    assert obs["joint_positions"] == [0.1, 0.2, 0.3]
    assert obs["joint_velocities"] == [0.01, 0.02, 0.03]
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_send_action_without_start_is_noop(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.send_action(
        {
            "joint_positions": [0.0] * 6,
            "joint_velocities": [0.0] * 6,
            "gripper_position": 0.0,
        }
    )


@pytest.mark.anyio
async def test_ros2_bridge_sim_control_without_start_is_noop(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.sim_control("play")


@pytest.mark.anyio
async def test_ros2_bridge_send_action_publishes_joint_state(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_publisher = bridge._publisher
    await bridge.send_action(
        {
            "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "joint_velocities": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "gripper_position": 0.8,
        }
    )

    assert mock_publisher.publish.call_count >= 1
    published_msg = mock_publisher.publish.call_args_list[0][0][0]
    assert len(published_msg.position) == 8
    assert published_msg.position[:7] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert published_msg.position[7] == 0.8
    assert len(published_msg.velocity) == 8
    assert published_msg.velocity[:7] == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    assert str(published_msg.velocity[7]) == "nan"

    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_send_action_publishes_float_array(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    config = ROS2Config(cameras={"wrist": "/cam/wrist"}, state_msg_type="Float32MultiArray")
    bridge = ROS2Bridge(config)
    await bridge.start()

    mock_publisher = bridge._float_array_publisher
    await bridge.send_action(
        {
            "joint_positions": [0.01, -0.02, 0.03, 0.1, -0.1, 0.05, 0.04],
            "joint_velocities": [float("nan")] * 7,
            "gripper_position": 0.04,
        }
    )

    assert mock_publisher.publish.call_count >= 1
    published_msg = mock_publisher.publish.call_args_list[0][0][0]
    # Float32MultiArray sends joint_positions directly — no separate gripper appended
    assert len(published_msg.data) == 7
    assert published_msg.data == [0.01, -0.02, 0.03, 0.1, -0.1, 0.05, 0.04]

    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_status_connecting_after_start(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    assert bridge.bridge_status == "connecting"
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_transitions_to_connected_on_observation(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    assert bridge.bridge_status == "connecting"

    bridge._on_joint_state_received([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
    assert bridge.bridge_status == "connected"
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_accepts_bridge_config(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import BridgeConfig, ROS2Config

    bridge_config = BridgeConfig(
        watchdog_timeout=5.0,
        reconnect_delay=1.0,
        max_reconnect_delay=15.0,
    )
    bridge = ROS2Bridge(ROS2Config(), bridge_config)
    assert bridge._watchdog_timeout == 5.0
    assert bridge._reconnect_delay == 1.0
    assert bridge._max_reconnect_delay == 15.0


@pytest.mark.anyio
async def test_ros2_bridge_get_latest_observation_returns_none_before_callbacks(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    obs = await bridge.get_latest_observation()
    assert obs is None
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_get_latest_observation_returns_data_after_callbacks(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    bridge._enqueue_interval = 0

    bridge._on_joint_state_received([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
    image_data = np.zeros((240, 320, 3), dtype=np.uint8)
    bridge._on_image_received("wrist", image_data)

    obs = await bridge.get_latest_observation()
    assert obs is not None
    assert "wrist" in obs["cameras"]
    assert obs["joint_positions"] == [0.1, 0.2, 0.3]
    assert obs["joint_velocities"] == [0.0, 0.0, 0.0]
    await bridge.close()


@pytest.mark.anyio
async def test_ros2_bridge_watchdog_cancels_on_close(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()
    assert bridge._watchdog_task is not None
    assert not bridge._watchdog_task.done()
    await bridge.close()
    assert bridge._watchdog_task is None


# ===== Priority 1: _call_service_async Tests =====


@pytest.mark.anyio
async def test_call_service_async_success(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = MagicMock()
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK

    # Schedule the callback to fire after executor runs
    async def trigger_callback():
        await asyncio.sleep(0.05)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_callback())  # noqa: RUF006

    mock_request = MagicMock()

    def request_factory():
        return mock_request

    result = await bridge._call_service_async(mock_client, request_factory, timeout=1.0)
    assert result is mock_response
    mock_client.call_async.assert_called_once_with(mock_request)
    await bridge.close()


@pytest.mark.anyio
async def test_call_service_async_timeout(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = MagicMock()
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    # Never call set_result — future never completes
    mock_request = MagicMock()

    def request_factory():
        return mock_request

    with pytest.raises(TimeoutError, match=r"Service call timed out after 0\.1s"):
        await bridge._call_service_async(mock_client, request_factory, timeout=0.1)

    await bridge.close()


@pytest.mark.anyio
async def test_call_service_async_exception(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = MagicMock()
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    test_exception = RuntimeError("Service failed")

    async def trigger_exception():
        await asyncio.sleep(0.05)
        fake_future.set_exception(test_exception)

    asyncio.create_task(trigger_exception())  # noqa: RUF006

    mock_request = MagicMock()

    def request_factory():
        return mock_request

    with pytest.raises(RuntimeError, match="Service failed"):
        await bridge._call_service_async(mock_client, request_factory, timeout=1.0)

    await bridge.close()


@pytest.mark.anyio
async def test_call_service_async_late_callback_after_timeout(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = MagicMock()
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_request = MagicMock()

    def request_factory():
        return mock_request

    # Future times out first
    with pytest.raises(TimeoutError):
        await bridge._call_service_async(mock_client, request_factory, timeout=0.1)

    # Then callback fires late — should not crash (already-done guard)
    mock_response = MagicMock()
    fake_future.set_result(mock_response)

    await bridge.close()


# ===== Priority 2: sim_control Tests =====


@pytest.mark.anyio
async def test_sim_control_play_success(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    # Configure mock service client
    mock_client = bridge._set_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK

    # Trigger callback shortly after call
    async def trigger_success():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_success())  # noqa: RUF006

    await bridge.sim_control("play")

    assert bridge._sim_paused is False
    mock_client.call_async.assert_called_once()
    request = mock_client.call_async.call_args[0][0]
    assert request.state.state == 1  # STATE_PLAYING
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_pause_success(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = bridge._set_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK

    async def trigger_success():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_success())  # noqa: RUF006

    await bridge.sim_control("pause")

    assert bridge._sim_paused is True
    request = mock_client.call_async.call_args[0][0]
    assert request.state.state == 2  # STATE_PAUSED
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_stop_success(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = bridge._set_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK

    async def trigger_success():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_success())  # noqa: RUF006

    await bridge.sim_control("stop")

    assert bridge._sim_paused is True
    request = mock_client.call_async.call_args[0][0]
    assert request.state.state == 0  # STATE_STOPPED
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_set_state_failure(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = bridge._set_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OPERATION_FAILED

    async def trigger_failure():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_failure())  # noqa: RUF006

    with pytest.raises(RuntimeError, match="SetSimulationState\\(play\\) failed"):
        await bridge.sim_control("play")

    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_set_state_timeout(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = bridge._set_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future
    # Never call set_result — timeout

    with pytest.raises(TimeoutError):
        await bridge.sim_control("play")

    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_step_success(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    config = ROS2Config(cameras={"wrist": "/cam/wrist"}, physics_decimation=10)
    bridge = ROS2Bridge(config)
    await bridge.start()

    mock_client = bridge._step_simulation_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK

    async def trigger_success():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_success())  # noqa: RUF006

    await bridge.sim_control("step")

    mock_client.call_async.assert_called_once()
    request = mock_client.call_async.call_args[0][0]
    assert request.steps == 10
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_step_failure(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    mock_client = bridge._step_simulation_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OPERATION_FAILED

    async def trigger_failure():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_failure())  # noqa: RUF006

    with pytest.raises(RuntimeError, match="StepSimulation failed"):
        await bridge.sim_control("step")

    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_reset_success(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    # Advance step counter first
    bridge._step = 42

    mock_client = bridge._reset_simulation_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK

    async def trigger_success():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_success())  # noqa: RUF006

    await bridge.sim_control("reset")

    assert bridge._step == 0
    mock_client.call_async.assert_called_once()
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_reset_preserves_step_on_failure(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    bridge._step = 42

    mock_client = bridge._reset_simulation_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OPERATION_FAILED

    async def trigger_failure():
        await asyncio.sleep(0.01)
        fake_future.set_result(mock_response)

    asyncio.create_task(trigger_failure())  # noqa: RUF006

    with pytest.raises(RuntimeError, match="ResetSimulation failed"):
        await bridge.sim_control("reset")

    assert bridge._step == 42  # Unchanged
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_reset_timeout(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    bridge._step = 42

    mock_client = bridge._reset_simulation_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future
    # Never call set_result

    with pytest.raises(TimeoutError):
        await bridge.sim_control("reset")

    assert bridge._step == 42  # Unchanged
    await bridge.close()


@pytest.mark.anyio
async def test_sim_control_unknown_action_is_noop(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    # Unknown action should not raise, just return
    await bridge.sim_control("unknown_action")

    # No service calls made
    assert not bridge._set_simulation_state_client.call_async.called
    assert not bridge._step_simulation_client.call_async.called
    assert not bridge._reset_simulation_client.call_async.called
    await bridge.close()


# ===== Priority 3: sim_state Property Tests =====


@pytest.mark.anyio
async def test_sim_state_none_returns_idle(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    assert bridge._sim_state is None
    assert bridge.sim_state == "idle"


@pytest.mark.anyio
async def test_sim_state_stopped_returns_idle(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    bridge._sim_state = 0  # STATE_STOPPED
    assert bridge.sim_state == "idle"


@pytest.mark.anyio
async def test_sim_state_playing_returns_running(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    bridge._sim_state = 1  # STATE_PLAYING
    assert bridge.sim_state == "running"


@pytest.mark.anyio
async def test_sim_state_paused_returns_paused(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    bridge._sim_state = 2  # STATE_PAUSED
    assert bridge.sim_state == "paused"


@pytest.mark.anyio
async def test_sim_state_unknown_returns_idle(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    bridge._sim_state = 99  # Unknown state
    assert bridge.sim_state == "idle"


# ===== Priority 3: _query_simulation_state Tests =====


@pytest.mark.anyio
async def test_query_state_no_client_is_noop(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    # Set client to None to test early exit
    bridge._get_simulation_state_client = None

    await bridge._query_simulation_state()
    # Should return without error
    await bridge.close()


@pytest.mark.anyio
async def test_query_state_first_query_sets_state(mock_rclpy, caplog):
    import logging

    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    assert bridge._sim_state is None

    mock_client = bridge._get_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK
    mock_response.state.state = 1  # STATE_PLAYING

    # Trigger callback immediately (fire-and-forget pattern)
    fake_future.set_result(mock_response)

    with caplog.at_level(logging.INFO):
        await bridge._query_simulation_state()
        await asyncio.sleep(0.05)  # Let callback process

    assert bridge._sim_state == 1
    # First query, no change log (old_state is None)
    assert "Simulation state changed" not in caplog.text
    await bridge.close()


@pytest.mark.anyio
async def test_query_state_unchanged_no_log(mock_rclpy, caplog):
    import logging

    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    bridge._sim_state = 1  # Already PLAYING

    mock_client = bridge._get_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK
    mock_response.state.state = 1  # Still PLAYING

    fake_future.set_result(mock_response)

    with caplog.at_level(logging.INFO):
        await bridge._query_simulation_state()
        await asyncio.sleep(0.05)

    assert bridge._sim_state == 1
    # State unchanged, no log
    assert "Simulation state changed" not in caplog.text
    await bridge.close()


@pytest.mark.anyio
async def test_query_state_changed_logs_transition(mock_rclpy, caplog):
    import logging

    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    bridge._sim_state = 1  # PLAYING

    mock_client = bridge._get_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    mock_response = MagicMock()
    mock_response.result.result = FakeResult.RESULT_OK
    mock_response.state.state = 2  # PAUSED

    fake_future.set_result(mock_response)

    with caplog.at_level(logging.INFO):
        await bridge._query_simulation_state()
        await asyncio.sleep(0.05)

    assert bridge._sim_state == 2
    # State changed, should log transition
    assert "Simulation state changed" in caplog.text
    assert "PLAYING -> PAUSED" in caplog.text
    await bridge.close()


@pytest.mark.anyio
async def test_query_state_exception_preserves_state(mock_rclpy):
    from robotics_playground.bridges.ros2_bridge import ROS2Bridge
    from robotics_playground.config import ROS2Config

    bridge = ROS2Bridge(ROS2Config(cameras={"wrist": "/cam/wrist"}))
    await bridge.start()

    bridge._sim_state = 1  # PLAYING

    mock_client = bridge._get_simulation_state_client
    fake_future = FakeRos2Future()
    mock_client.call_async.return_value = fake_future

    test_exception = RuntimeError("Service failed")
    fake_future.set_exception(test_exception)

    await bridge._query_simulation_state()
    await asyncio.sleep(0.05)

    # State unchanged on exception
    assert bridge._sim_state == 1
    await bridge.close()
