from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from robotics_playground.config import ModelConfig, PolicyConfig
from robotics_playground.session import Session


class PresetChunkSessionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self):
        bridge = MagicMock()
        bridge.execution_status = SimpleNamespace(
            paused=False,
            idle=True,
            remaining=0,
        )
        bridge.get_observation = AsyncMock(return_value={
            "step": 1,
            "cameras": {},
            "state": [0.0] * 8,
            "joint_positions": [],
            "joint_velocities": [],
        })
        bridge.send_action = AsyncMock()
        bridge.execute_action_chunk = AsyncMock()
        bridge.sim_control = AsyncMock()

        session = Session(
            bridge=bridge,
            policy_config=PolicyConfig(),
            rerun_logger=MagicMock(),
            preset=MagicMock(),
        )

        adapter = MagicMock()
        chunk = np.full((15, 8), 0.25, dtype=np.float32)
        adapter.observation_to_openpi.return_value = {"prompt": "test"}
        adapter.action_chunk_for_environment.return_value = chunk

        policy = MagicMock()
        policy.infer = AsyncMock(return_value={"actions": chunk})

        session._adapter = adapter
        session._policy = policy
        session._policy_status = "connected"
        session.sim_state = "running"
        session.set_instruction("test")

        return session, bridge, adapter, policy, chunk

    async def stop_task(self, task):
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_waits_for_chunk_completion_without_single_actions(self):
        session, bridge, adapter, policy, chunk = self.make_session()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def execute(values, timeout):
            np.testing.assert_array_equal(values, chunk)
            entered.set()
            await release.wait()
            session.clear_instruction()

        bridge.execute_action_chunk.side_effect = execute
        task = asyncio.create_task(session._inference_loop())

        try:
            await asyncio.wait_for(entered.wait(), timeout=1.0)

            self.assertEqual(policy.infer.await_count, 1)
            bridge.send_action.assert_not_awaited()
            adapter.action_chunk_from_openpi.assert_not_called()

            release.set()
            for _ in range(5):
                await asyncio.sleep(0)

            self.assertEqual(policy.infer.await_count, 1)
            self.assertFalse(task.done())
        finally:
            await self.stop_task(task)

    async def test_missing_actions_pauses_without_killing_loop(self):
        session, bridge, _, policy, _ = self.make_session()
        policy.infer.return_value = {"unexpected": []}
        paused = asyncio.Event()

        async def pause(action):
            self.assertEqual(action, "pause")
            paused.set()

        bridge.sim_control.side_effect = pause
        task = asyncio.create_task(session._inference_loop())

        try:
            await asyncio.wait_for(paused.wait(), timeout=1.0)
            self.assertEqual(session.sim_state, "paused")
            self.assertFalse(task.done())
            bridge.execute_action_chunk.assert_not_awaited()
            bridge.send_action.assert_not_awaited()
        finally:
            await self.stop_task(task)

    async def test_discards_response_after_pause_during_inference(self):
        session, bridge, adapter, policy, chunk = self.make_session()
        response_returned = asyncio.Event()

        async def infer(request):
            session.sim_state = "paused"
            asyncio.get_running_loop().call_soon(response_returned.set)
            return {"actions": chunk}

        policy.infer.side_effect = infer
        task = asyncio.create_task(session._inference_loop())

        try:
            await asyncio.wait_for(response_returned.wait(), timeout=1.0)

            self.assertFalse(task.done())
            bridge.execute_action_chunk.assert_not_awaited()
            bridge.send_action.assert_not_awaited()
            adapter.action_chunk_for_environment.assert_not_called()
        finally:
            await self.stop_task(task)

    async def test_reset_cancels_inference_before_reset_service(self):
        session, bridge, _, policy, _ = self.make_session()
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def infer(request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def reset(action, speed=None):
            self.assertEqual(action, "reset")
            self.assertTrue(cancelled.is_set())

        policy.infer.side_effect = infer
        bridge.sim_control.side_effect = reset

        await session.start_loop()

        try:
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            await asyncio.wait_for(
                session.sim_control("reset"),
                timeout=1.0,
            )

            self.assertEqual(session.sim_state, "idle")
            self.assertEqual(session.instruction, "")
            bridge.execute_action_chunk.assert_not_awaited()
            bridge.send_action.assert_not_awaited()
        finally:
            await session.stop_loop()

    async def test_model_switch_cancels_pauses_and_clears_before_connect(self):
        session, bridge, _, policy, _ = self.make_session()
        entered = asyncio.Event()
        order = []

        session._preset.id = "test_preset"
        session._preset.action.dim = 8
        session._model_id = "old_model"
        session._policy_config = PolicyConfig(models={
            "new_model": ModelConfig(
                endpoint="ws://unused",
                preset_id="test_preset",
            ),
        })

        async def infer(request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                order.append("cancel")

        async def pause(action):
            self.assertEqual(action, "pause")
            order.append("pause")

        async def clear(chunk):
            self.assertEqual(chunk.shape, (0, 8))
            order.append("clear")

        async def close():
            order.append("close")

        policy.infer.side_effect = infer
        policy.close = AsyncMock(side_effect=close)
        bridge.sim_control.side_effect = pause
        bridge.send_action_chunk = AsyncMock(side_effect=clear)
        session._start_policy_connect = MagicMock(
            side_effect=lambda: order.append("connect")
        )

        await session.start_loop()

        try:
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            await asyncio.wait_for(
                session.select_model("new_model"),
                timeout=1.0,
            )

            self.assertEqual(
                order,
                ["cancel", "pause", "clear", "close", "connect"],
            )
            self.assertEqual(session.model_id, "new_model")
            self.assertEqual(session.sim_state, "paused")
            self.assertEqual(session.instruction, "")
            bridge.execute_action_chunk.assert_not_awaited()
        finally:
            await session.stop_loop()

if __name__ == "__main__":
    unittest.main()
