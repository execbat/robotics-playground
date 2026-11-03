from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def mock_rr():
    with patch("robotics_playground.rerun_logger.rr") as mock:
        yield mock


def test_log_observation_logs_all_cameras(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.flush()
    mock_rr.reset_mock()

    obs = {
        "step": 5,
        "cameras": {
            "wrist": np.zeros((240, 320, 3), dtype=np.uint8),
            "head": np.zeros((240, 320, 3), dtype=np.uint8),
        },
        "joint_positions": [0.1, 0.2, 0.3],
        "joint_velocities": [0.0, 0.0, 0.0],
    }
    logger.log_observation(obs, step=5, wallclock_time=100.0)
    logger.flush()

    mock_rr.set_time.assert_any_call("step", sequence=5)
    mock_rr.set_time.assert_any_call("time", duration=0.0)  # First log sets start_time

    logged_paths = [call.args[0] for call in mock_rr.log.call_args_list]
    assert "session/policy_0/camera/wrist" in logged_paths
    assert "session/policy_0/camera/head" in logged_paths
    logger.shutdown()


def test_log_observation_logs_joint_positions(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.flush()
    mock_rr.reset_mock()

    obs = {
        "step": 0,
        "cameras": {"wrist": np.zeros((240, 320, 3), dtype=np.uint8)},
        "joint_positions": [0.1, 0.2, 0.3],
        "joint_velocities": [0.4, 0.5, 0.6],
    }
    logger.log_observation(obs, step=0)
    logger.flush()

    logged_paths = [call.args[0] for call in mock_rr.log.call_args_list]
    assert "session/policy_0/joints/shoulder_rot" in logged_paths
    assert "session/policy_0/joints/elbow_rot" in logged_paths
    logger.shutdown()


def test_log_action_logs_dimensions(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.flush()
    logger._start_time = 10.0  # Set baseline
    mock_rr.reset_mock()

    action = {"joint_positions": [0.1, 0.2, 0.3]}
    logger.log_action(action, step=1, wallclock_time=11.5)
    logger.flush()

    mock_rr.set_time.assert_any_call("step", sequence=1)
    mock_rr.set_time.assert_any_call("time", duration=1.5)  # 11.5 - 10.0
    logged_paths = [call.args[0] for call in mock_rr.log.call_args_list]
    assert "session/policy_0/actions/dim_0" in logged_paths
    logger.shutdown()


def test_log_instruction(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.flush()
    logger._start_time = 10.0  # Set baseline
    mock_rr.reset_mock()

    logger.log_instruction("pick up block", step=3, wallclock_time=12.0)
    logger.flush()

    mock_rr.set_time.assert_any_call("step", sequence=3)
    mock_rr.set_time.assert_any_call("time", duration=2.0)  # 12.0 - 10.0
    mock_rr.log.assert_called()
    logger.shutdown()


def test_clear_logs_clear_markers(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger(port=9876)
    logger.start()

    try:
        logger.flush()
        mock_rr.reset_mock()

        logger._last_step = 10
        logger.clear()
        logger.flush()

        # Clear is placed after the previous data.
        mock_rr.set_time.assert_any_call("step", sequence=11)
        mock_rr.Clear.assert_any_call(recursive=True)

        mock_rr.log.assert_any_call(
            "session/policy_0",
            mock_rr.Clear.return_value,
        )
        mock_rr.log.assert_any_call(
            "session/instructions",
            mock_rr.Clear.return_value,
        )

        # New data must start after the clear marker.
        assert logger._step_offset == 12
        assert logger._last_step == 0
        assert logger._start_time is None
    finally:
        logger.shutdown()


def test_clear_advances_step_offset():
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger._initialized = True
    logger._step_offset = 100
    logger._last_step = 10

    logger.clear()

    # Previous effective step: 110.
    # Clear marker: 111. Next episode starts at 112.
    assert logger._step_offset == 112
    assert logger._last_step == 0


def test_clear_resets_start_time(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger._initialized = True
    logger._start_time = 100.0
    logger.clear()
    assert logger._start_time is None


def test_log_after_clear_starts_after_previous_episode(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()

    try:
        logger.flush()

        logger._step_offset = 100
        logger._last_step = 5
        logger.clear()
        logger.flush()

        # Previous data ends at 105; clear is recorded at 106.
        mock_rr.set_time.assert_any_call("step", sequence=106)
        assert logger._step_offset == 107

        mock_rr.reset_mock()

        obs = {
            "step": 0,
            "cameras": {
                "wrist": np.zeros((240, 320, 3), dtype=np.uint8),
            },
            "joint_positions": [0.1],
            "joint_velocities": [0.0],
        }

        logger.log_observation(obs, step=0, wallclock_time=25.0)
        logger.flush()

        # Environment step 0 maps to recording step 107.
        mock_rr.set_time.assert_any_call("step", sequence=107)

        # Elapsed time starts from the new episode's baseline.
        mock_rr.set_time.assert_any_call("time", duration=0.0)

        assert logger._last_step == 0
        assert logger._start_time == 25.0
    finally:
        logger.shutdown()


def test_clear_before_start_is_noop(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.clear()
    mock_rr.log.assert_not_called()


def test_log_raw_action_tensor_no_error(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    actions = np.zeros((10, 8), dtype=np.float32)
    logger.log_raw_action_tensor(actions, step=0)
    logger.flush()
    logger.shutdown()


def test_log_inference_latency_no_error(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.log_inference_latency(42.5, step=0)
    logger.flush()
    logger.shutdown()


def test_log_action_trajectory_no_error(mock_rr):
    import math

    from robotics_playground.bridges.protocol import Action
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    chunk = [
        Action(
            joint_positions=[0.1 * i] * 7,
            joint_velocities=[math.nan] * 7,
            gripper_position=0.02,
        )
        for i in range(10)
    ]
    logger.log_action_trajectory(chunk, step=0)
    logger.flush()
    logger.shutdown()


def test_log_observation_does_not_block_caller(mock_rr):
    """rr.log calls run on worker thread, not on the calling thread."""
    from robotics_playground.rerun_logger import RerunLogger

    caller_thread = threading.current_thread().ident
    log_threads: list[int] = []

    def capture_thread(*args, **kwargs):
        log_threads.append(threading.current_thread().ident)

    logger = RerunLogger()
    logger.start()
    logger.flush()

    mock_rr.log.side_effect = capture_thread

    obs = {
        "step": 0,
        "cameras": {"wrist": np.zeros((240, 320, 3), dtype=np.uint8)},
        "joint_positions": [0.1, 0.2],
        "joint_velocities": [0.0, 0.0],
    }
    logger.log_observation(obs, step=0)
    logger.flush()
    logger.shutdown()

    assert len(log_threads) > 0
    for tid in log_threads:
        assert tid != caller_thread, "rr.log must not run on the calling thread"


def test_log_observation_returns_immediately_when_rr_is_slow(mock_rr):
    """Caller returns in <50ms even when rr.log takes 2 seconds."""
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.flush()

    mock_rr.log.side_effect = lambda *a, **kw: time.sleep(2.0)

    obs = {
        "step": 0,
        "cameras": {"wrist": np.zeros((240, 320, 3), dtype=np.uint8)},
        "joint_positions": [0.1],
        "joint_velocities": [0.0],
    }
    t0 = time.monotonic()
    logger.log_observation(obs, step=0)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.05, f"log_observation blocked for {elapsed:.3f}s"
    logger.shutdown()


def test_flush_drains_queue(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    logger.flush()
    mock_rr.reset_mock()

    obs = {
        "step": 0,
        "cameras": {"wrist": np.zeros((240, 320, 3), dtype=np.uint8)},
        "joint_positions": [0.1],
        "joint_velocities": [0.0],
    }
    logger.log_observation(obs, step=0)
    logger.log_observation(obs, step=1)
    logger.flush()

    assert mock_rr.log.call_count >= 2
    logger.shutdown()


def test_shutdown_stops_worker(mock_rr):
    from robotics_playground.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.start()
    worker = logger._worker_thread
    assert worker is not None
    assert worker.is_alive()

    logger.shutdown()
    assert not worker.is_alive()
