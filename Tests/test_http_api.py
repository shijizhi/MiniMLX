import json
import socket
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from minimlx_runtime.api.http import (
    DEFAULT_API_MAX_TOKENS,
    _ApiTelemetry,
    _DualStackThreadingHTTPServer,
    _anthropic_request,
    _apply_model_defaults,
    _finish_reason,
    _infer,
    _message_content,
    _model_id,
    _prometheus_metrics,
)
from minimlx_runtime.api.memory import (
    GIB,
    InferenceAdmission,
    MemorySnapshot,
    SystemMemoryMonitor,
    admission_error,
    estimate_context_memory_bytes,
    max_parallel_requests,
    memory_reserve_bytes,
    read_system_memory,
)
from minimlx_runtime.types import ContentPart, InferenceRequest, InputMessage, ModelTask, Modality


class HttpApiTests(unittest.TestCase):
    def test_default_memory_reserve_matches_omlx_aggressive_tier(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(memory_reserve_bytes(64 * GIB), GIB + GIB // 2)
            self.assertEqual(
                memory_reserve_bytes(128 * GIB), 2 * GIB + GIB * 56 // 100
            )

    def test_dual_stack_listener_accepts_ipv4_mapped_connections(self):
        if not socket.has_ipv6:
            self.skipTest("IPv6 is unavailable")

        server = _DualStackThreadingHTTPServer(("::", 0), object, bind_and_activate=False)
        try:
            with patch.object(socket.socket, "bind", return_value=None):
                server.server_bind()
            self.assertEqual(server.socket.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY), 0)
        finally:
            server.server_close()

    def test_memory_admission_refuses_only_when_system_reserve_is_breached(self):
        total = 32 * GIB
        reserve = memory_reserve_bytes(total)
        self.assertIsNone(admission_error(MemorySnapshot(total, reserve, total - reserve, "test")))
        reason = admission_error(
            MemorySnapshot(total, reserve - 1, total - reserve + 1, "test")
        )
        self.assertIn("active response was not interrupted", reason)
        self.assertIn("retry the request", reason)

    def test_inference_admission_rejects_overlap_and_preserves_active_slot(self):
        snapshot = MemorySnapshot(32 * GIB, 16 * GIB, 16 * GIB, "test")
        admission = InferenceAdmission(1, memory_reader=lambda: snapshot)
        current = admission.acquire()
        self.assertTrue(current.allowed)
        rejected = admission.acquire()
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.code, "server_busy")
        admission.release()
        self.assertTrue(admission.acquire().allowed)
        admission.release()

    def test_inference_admission_rejects_low_or_unknown_memory_without_leaking_slot(self):
        total = 32 * GIB
        low_memory = MemorySnapshot(total, 1, total - 1, "test")
        admission = InferenceAdmission(1, memory_reader=lambda: low_memory)
        rejected = admission.acquire()
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.code, "memory_pressure")

        unknown = InferenceAdmission(
            1, memory_reader=lambda: MemorySnapshot(None, None, None, "unavailable")
        )
        rejected = unknown.acquire()
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.code, "memory_check_unavailable")

    def test_request_admission_includes_configured_context_cache_headroom(self):
        runtime = SimpleNamespace(
            descriptor=SimpleNamespace(metadata={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            })
        )
        context_cache = estimate_context_memory_bytes(runtime, 8192)
        self.assertEqual(context_cache, 8192 * 32 * 8 * 128 * 4)

        total = 32 * GIB
        admission = InferenceAdmission(
            1,
            memory_reader=lambda: MemorySnapshot(total, 4 * GIB, 28 * GIB, "test"),
            additional_reserve_bytes=context_cache,
        )
        self.assertEqual(admission.acquire().code, "memory_pressure")

    def test_memory_admission_fails_closed_when_platform_metrics_are_unavailable(self):
        reason = admission_error(MemorySnapshot(None, None, None, "unavailable"))
        self.assertIn("could not verify available system memory", reason)

    def test_memory_monitor_tracks_pressure_and_recovery_without_stopping_inference(self):
        from unittest.mock import patch
        from minimlx_runtime.api import memory

        total = 32 * GIB
        reserve = memory_reserve_bytes(total)
        monitor = SystemMemoryMonitor()
        with patch.object(
            memory,
            "read_system_memory",
            return_value=MemorySnapshot(total, reserve - 1, total - reserve + 1, "test"),
        ):
            monitor._sample()
        self.assertEqual(monitor.snapshot()["state"], "reserve")
        self.assertEqual(monitor.snapshot()["peak_used_bytes"], total - reserve + 1)

        with patch.object(
            memory,
            "read_system_memory",
            return_value=MemorySnapshot(total, reserve, total - reserve, "test"),
        ):
            monitor._sample()
        self.assertEqual(monitor.snapshot()["state"], "normal")

    def test_max_parallel_requests_defaults_to_single_flight_and_is_bounded(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(max_parallel_requests(), 1)
        with patch.dict("os.environ", {"MINIMLX_MAX_PARALLEL_REQUESTS": "99"}):
            self.assertEqual(max_parallel_requests(), 8)
        with patch.dict("os.environ", {"MINIMLX_MAX_PARALLEL_REQUESTS": "invalid"}):
            self.assertEqual(max_parallel_requests(), 1)

    def test_darwin_memory_snapshot_uses_reclaimable_inactive_pages_as_available(self):
        from unittest.mock import patch
        from minimlx_runtime.api import memory

        vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages active: 20.\nPages inactive: 30.\nPages wired down: 5.\nPages occupied by compressor: 2.\n"""
        with (
            patch.object(memory.platform, "system", return_value="Darwin"),
            patch.object(memory, "_command_text", side_effect=[str(64 * GIB), vm_stat]),
        ):
            snapshot = read_system_memory()
        expected_used = (20 + 5 + 2) * 16384
        self.assertEqual(snapshot.used_bytes, expected_used)
        self.assertEqual(snapshot.available_bytes, 64 * GIB - expected_used)

    def test_darwin_memory_snapshot_falls_back_to_sysconf_when_sysctl_is_blocked(self):
        from unittest.mock import patch
        from minimlx_runtime.api import memory

        values = {"SC_PHYS_PAGES": 16, "SC_PAGESIZE": 4096}
        with (
            patch.object(memory.platform, "system", return_value="Darwin"),
            patch.object(memory, "_command_text", side_effect=[None, "Pages active: 1."]),
            patch.object(memory.os, "sysconf", side_effect=lambda key: values[key]),
        ):
            snapshot = read_system_memory()
        self.assertEqual(snapshot.total_bytes, 16 * 4096)
        self.assertEqual(snapshot.source, "darwin-vm_stat")

    def test_force_sampling_replaces_client_sampler_values_with_model_defaults(self):
        body = {"temperature": 0.0, "top_p": 0.1, "max_tokens": 50}
        _apply_model_defaults(
            body,
            {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
            force_sampling=True,
        )
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["top_k"], 40)
        self.assertEqual(body["max_tokens"], 50)

    def test_request_limits_cannot_exceed_service_context_and_prefill_is_capped(self):
        with self.assertRaisesRegex(ValueError, "exceeds the configured service context limit"):
            _apply_model_defaults(
                {"messages": [], "max_context_tokens": 16384},
                {"max_context_tokens": 8192},
                force_sampling=False,
            )
        with self.assertRaisesRegex(ValueError, "max_tokens.*exceeds"):
            _apply_model_defaults(
                {"messages": [], "max_tokens": 9000},
                {"max_context_tokens": 8192},
                force_sampling=False,
            )

        body = {"messages": [], "prefill_step_size": 8192}
        _apply_model_defaults(
            body,
            {"max_context_tokens": 8192},
            force_sampling=False,
            default_max_tokens=32768,
        )
        self.assertEqual(body["prefill_step_size"], 2048)
        self.assertEqual(body["max_tokens"], 8192)

    def test_chat_template_parameters_are_forwarded_to_runtime(self):
        class FakeRuntime:
            def infer(self, request):
                return SimpleNamespace(payload=SimpleNamespace(text="reply"), usage={}, metrics={})

        request, _ = _infer(FakeRuntime(), {
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
        })
        self.assertEqual(request.options["chat_template_kwargs"], {"enable_thinking": False})

    def test_api_model_alias_overrides_local_checkpoint_directory_name(self):
        runtime = SimpleNamespace(descriptor=SimpleNamespace(model_dir="/models/publisher/checkpoint"))
        self.assertEqual(_model_id(runtime), "checkpoint")
        self.assertEqual(_model_id(runtime, "assistant-qwen"), "assistant-qwen")

    def test_content_accepts_text_and_local_or_data_uri_images(self):
        self.assertEqual(_message_content("hello")[0].text, "hello")
        image = _message_content([
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ])
        self.assertEqual(image[0].modality, Modality.TEXT)
        self.assertEqual(image[1].modality, Modality.IMAGE)
        self.assertTrue(image[1].uri.startswith("data:image/"))

    def test_content_accepts_timestamped_video_frame_data_uris(self):
        video = _message_content([{
            "type": "video",
            "frames": [
                {"timestamp_seconds": 0.0, "image_url": "data:image/png;base64,aW1hZ2Uw"},
                {"timestamp_seconds": 1.25, "image_url": "data:image/jpeg;base64,aW1hZ2Ux"},
            ],
        }])[0]
        self.assertEqual(video.modality, Modality.VIDEO)
        self.assertEqual([frame.timestamp_seconds for frame in video.frames], [0.0, 1.25])
        self.assertEqual(video.frames[1].mime_type, "image/jpeg")

    def test_openai_chat_request_maps_to_runtime_contract(self):
        class FakeRuntime:
            def infer(self, request):
                self.request = request
                return SimpleNamespace(
                    payload=SimpleNamespace(text="reply:hello"),
                    usage={"prompt_tokens": 3, "completion_tokens": 2},
                    metrics={"prefill_ms": 10.0, "decode_ms": 5.0},
                )

        runtime = FakeRuntime()
        request, result = _infer(runtime, {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 17,
        })
        self.assertEqual(request.task.value, "generation")
        self.assertEqual(request.messages[0].content[0].text, "hello")
        self.assertEqual(request.options["max_tokens"], 17)
        self.assertEqual(result.payload.text, "reply:hello")

    def test_openai_chat_defaults_max_tokens_to_omlx_default(self):
        class FakeRuntime:
            def infer(self, request):
                self.request = request
                return SimpleNamespace(
                    payload=SimpleNamespace(text="reply"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                    metrics={},
                )

        runtime = FakeRuntime()
        request, _ = _infer(runtime, {"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(DEFAULT_API_MAX_TOKENS, 32768)
        self.assertEqual(request.options["max_tokens"], 32768)

    def test_openai_chat_uses_configured_default_max_tokens(self):
        class FakeRuntime:
            def infer(self, request):
                return SimpleNamespace(
                    payload=SimpleNamespace(text="reply"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                    metrics={},
                )

        request, _ = _infer(
            FakeRuntime(), {"messages": [{"role": "user", "content": "hello"}]}, 4096
        )
        self.assertEqual(request.options["max_tokens"], 4096)

    def test_api_generation_options_are_forwarded_to_runtime(self):
        class FakeRuntime:
            def infer(self, request):
                return SimpleNamespace(payload=SimpleNamespace(text="reply"), usage={}, metrics={})

        request, _ = _infer(FakeRuntime(), {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.08,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.15,
            "max_context_tokens": 8192,
        })
        self.assertEqual(request.options["temperature"], 0.7)
        self.assertEqual(request.options["top_p"], 0.9)
        self.assertEqual(request.options["top_k"], 40)
        self.assertEqual(request.options["min_p"], 0.08)
        self.assertEqual(request.options["repetition_penalty"], 1.1)
        self.assertEqual(request.options["presence_penalty"], 0.2)
        self.assertEqual(request.options["frequency_penalty"], 0.15)
        self.assertEqual(request.options["max_context_tokens"], 8192)

    def test_max_completion_tokens_alias_is_preserved(self):
        class FakeRuntime:
            def infer(self, request):
                return SimpleNamespace(
                    payload=SimpleNamespace(text="reply"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                    metrics={},
                )

        request, _ = _infer(FakeRuntime(), {
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 256,
        })
        self.assertEqual(request.options["max_tokens"], 256)

    def test_finish_reason_reports_token_limit(self):
        request = InferenceRequest(
            task=ModelTask.GENERATION,
            messages=(
                InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),)),
            ),
            options={"max_tokens": 256},
        )
        self.assertEqual(_finish_reason(request, 256), "length")
        self.assertEqual(_finish_reason(request, 255), "stop")

    def test_anthropic_messages_maps_system_text_and_image_content(self):
        converted = _anthropic_request({
            "system": "Be concise",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is shown?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
            ]}],
        })
        self.assertEqual(converted["messages"][0], {"role": "system", "content": "Be concise"})
        parts = _message_content(converted["messages"][1]["content"])
        self.assertEqual(parts[0].text, "What is shown?")
        self.assertEqual(parts[1].data_base64, "aGVsbG8=")

    def test_prometheus_endpoint_uses_text_exposition_and_metric_types(self):
        exposition = _prometheus_metrics({
            "requests": 4, "active_requests": 1, "failed_requests": 2,
            "prompt_tokens": 100, "completion_tokens": 80,
            "prefill_tps": 20.0, "generation_tps": 10.0, "uptime_seconds": 5,
        })
        self.assertIn("# TYPE minimlx_requests_total counter", exposition)
        self.assertIn("# TYPE minimlx_requests_active gauge", exposition)
        self.assertIn("minimlx_prompt_tokens_total 100", exposition)

    def test_usage_history_persists_aggregate_request_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = f"{directory}/usage.sqlite3"
            telemetry = _ApiTelemetry(history_path)
            telemetry.finish(model="fixture", started=__import__("time").perf_counter(), prompt_tokens=8, completion_tokens=3)
            with sqlite3.connect(history_path) as database:
                row = database.execute("SELECT model, status, prompt_tokens, completion_tokens FROM usage_history").fetchone()
            self.assertEqual(row, ("fixture", "complete", 8, 3))

    def test_text_stream_uses_incremental_runtime_events(self):
        class FakeRuntime:
            def infer_stream(self, request):
                self.stream_flag = request.stream
                yield {"event": "delta", "delta": "first", "token_ids": [11]}
                yield {"event": "delta", "delta": " second", "token_ids": [12]}
                yield {
                    "event": "done", "text": "first second",
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                    "metrics": {"prefill_ms": 10.0, "decode_ms": 5.0},
                }

        runtime = FakeRuntime()
        request, events = _infer(runtime, {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        })
        events = list(events)
        self.assertTrue(request.stream)
        self.assertTrue(runtime.stream_flag)
        self.assertEqual([event["delta"] for event in events if event["event"] == "delta"], ["first", " second"])

    def test_telemetry_aggregates_counts_rates_and_recent_requests(self):
        telemetry = _ApiTelemetry()
        started = __import__("time").perf_counter()
        telemetry.begin()
        self.assertEqual(telemetry.snapshot()["active_requests"], 1)
        telemetry.finish(
            model="fixture",
            started=started,
            prompt_tokens=20,
            completion_tokens=10,
            prefill_ms=10,
            decode_ms=5,
            vision_cache_hits=3,
            vision_cache_misses=1,
        )
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["prompt_tokens"], 20)
        self.assertEqual(snapshot["completion_tokens"], 10)
        self.assertEqual(snapshot["prefill_tps"], 2000.0)
        self.assertEqual(snapshot["generation_tps"], 2000.0)
        self.assertEqual(snapshot["vision_cache_hit_rate"], 75.0)
        self.assertEqual(snapshot["recent_requests"][0]["model"], "fixture")

    def test_telemetry_tracks_busy_and_memory_rejections_separately(self):
        telemetry = _ApiTelemetry()
        telemetry.record_rejection("busy")
        telemetry.record_rejection("memory")
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["rejected_busy_requests"], 1)
        self.assertEqual(snapshot["rejected_memory_requests"], 1)


if __name__ == "__main__":
    unittest.main()
