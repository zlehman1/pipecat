#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute.function_call_user_mute_strategy import FunctionCallUserMuteStrategy


class MockLLMService(LLMService):
    """Minimal LLM service for testing function call execution."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)


class PipelineTestLLMService(MockLLMService):
    """Mock LLM service that lets pipeline frames reach the test sink."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class FunctionCallTriggerAndRecordingProcessor(FrameProcessor):
    """Starts a test function call and captures downstream frames."""

    def __init__(self, service: MockLLMService):
        super().__init__(enable_direct_mode=True)
        self._service = service
        self.downstream_frames: list[Frame] = []
        self.function_call_cancelled = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.downstream_frames.append(frame)
            if isinstance(frame, FunctionCallCancelFrame):
                self.function_call_cancelled.set()
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.create_task(
                self._service.run_function_calls(
                    [
                        FunctionCallFromLLM(
                            function_name="slow_tool",
                            tool_call_id="call_1",
                            arguments={},
                            context=LLMContext(),
                        )
                    ]
                )
            )


class TestLLMService(unittest.IsolatedAsyncioTestCase):
    async def _run_function_calls_inline(self, service: MockLLMService):
        async def run_inline(runner_items):
            for runner_item in runner_items:
                await service._run_function_call(runner_item)

        service._run_parallel_function_calls = run_inline
        service._run_sequential_function_calls = run_inline

    async def test_missing_function_call_emits_terminal_result(self):
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        with patch("pipecat.services.llm_service.logger") as mock_logger:
            await service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="missing_tool",
                        tool_call_id="call_1",
                        arguments={"query": "weather"},
                        context=LLMContext(),
                    )
                ]
            )

        self.assertEqual(
            [type(frame) for frame in recorded_frames],
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
            ],
        )
        self.assertEqual(recorded_frames[1].function_name, "missing_tool")
        self.assertEqual(
            recorded_frames[2].result,
            "Error: function 'missing_tool' is not registered.",
        )

        # Only the queue-time warning should fire; the execution-time
        # "just unregistered" warning must not double-log.
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("not registered" in w for w in warnings))
        self.assertFalse(any("just unregistered" in w for w in warnings))

    async def test_function_unregistered_between_queue_and_execute(self):
        """Function unregistered between queuing and execution still terminates."""
        service = MockLLMService()
        service._call_event_handler = AsyncMock()

        async def real_handler(params):
            await params.result_callback("should not be called")

        service.register_function("doomed_tool", real_handler)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        async def run_inline(runner_items):
            # Simulate the function being unregistered after queuing but before execution.
            service.unregister_function("doomed_tool")
            for runner_item in runner_items:
                await service._run_function_call(runner_item)

        service._run_parallel_function_calls = run_inline
        service._run_sequential_function_calls = run_inline

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="doomed_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        self.assertEqual(
            [type(frame) for frame in recorded_frames],
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
            ],
        )
        self.assertEqual(
            recorded_frames[2].result,
            "Error: function 'doomed_tool' is not registered.",
        )

    async def test_missing_function_call_allows_user_mute_cleanup(self):
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="missing_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        strategy = FunctionCallUserMuteStrategy()
        muted = False
        for frame in recorded_frames:
            muted = await strategy.process_frame(frame)

        self.assertFalse(muted)

    async def test_muted_function_call_interruption_cancels_running_tool(self):
        service = PipelineTestLLMService()
        service._call_event_handler = AsyncMock()
        user_aggregator = LLMUserAggregator(
            LLMContext(),
            params=LLMUserAggregatorParams(
                user_mute_strategies=[FunctionCallUserMuteStrategy()]
            ),
        )
        recorder = FunctionCallTriggerAndRecordingProcessor(service)
        pipeline = Pipeline([user_aggregator, service, recorder])

        tool_started = asyncio.Event()
        tool_cancelled = asyncio.Event()
        tool_timed_out = asyncio.Event()

        async def slow_tool(params):
            tool_started.set()
            try:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=0.5)
                await params.result_callback("unexpected result")
            except asyncio.TimeoutError:
                tool_timed_out.set()
                await params.result_callback("not cancelled")
            except asyncio.CancelledError:
                tool_cancelled.set()
                raise

        service.register_function("slow_tool", slow_tool, cancel_on_interruption=True)

        await run_test(
            pipeline,
            frames_to_send=[
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                SleepFrame(sleep=0.05),
            ],
        )

        self.assertTrue(tool_started.is_set())
        self.assertTrue(tool_cancelled.is_set())
        self.assertTrue(recorder.function_call_cancelled.is_set())
        self.assertFalse(tool_timed_out.is_set())
        self.assertFalse(user_aggregator._user_is_muted)
        self.assertTrue(
            any(isinstance(frame, FunctionCallCancelFrame) for frame in recorder.downstream_frames)
        )
