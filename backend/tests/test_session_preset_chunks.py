from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from robotics_playground.config import PolicyConfig
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

if __name__ == "__main__":
    unittest.main()
