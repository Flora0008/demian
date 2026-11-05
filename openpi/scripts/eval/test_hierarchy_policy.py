"""CPU-only regression checks for the user-approved recovery and trace policy."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

from hierarchy_trace import EventLog
from test_hierarchical_controller import Backend, controller, observe, verdict, phase


class PolicyTests(unittest.TestCase):
    def test_two_failures_and_reset_only_after_confirmed_advance(self):
        backend = Backend(verdicts=[verdict("failure"), verdict("failure"),
                                    verdict("success"), verdict("success")])
        ctrl = controller(backend, confirmations=2)
        self.addCleanup(ctrl.close)
        ctrl.start({"frame": 0}, "task")
        observe(ctrl, {"frame": 1}, 1)
        observe(ctrl, {"frame": 2}, 2)
        self.assertEqual(ctrl.replans, 0)
        self.assertIsNone(ctrl.caption)
        observe(ctrl, {"frame": 3}, 3)
        self.assertEqual(ctrl.replans, 1)  # Installing repair does NOT reset budget.
        for step in (4, 5, 6):
            observe(ctrl, {"frame": step}, step)
        self.assertEqual(ctrl.instruction, "two")
        self.assertEqual(ctrl.replans, 0)
        self.assertEqual(ctrl.total_replans, 1)

    def test_two_repairs_are_executed_before_exhaustion(self):
        ctrl = controller(Backend(verdicts=[verdict("failure")] * 20), confirmations=2)
        self.addCleanup(ctrl.close)
        ctrl.start({"frame": 0}, "task")
        for step in range(1, 9):
            observe(ctrl, {"frame": step}, step)
            self.assertFalse(ctrl.stopped)
        observe(ctrl, {"frame": 9}, 9)
        self.assertEqual(ctrl.replans, 2)
        self.assertIn("stage_replan_budget_exhausted", ctrl.error)

    def test_transport_error_between_failures_resets_streak(self):
        backend = Backend()
        calls = 0
        def serve(op, payload):
            nonlocal calls
            if op == "verify":
                calls += 1
                if calls == 2:
                    raise TimeoutError("network")
                return verdict("failure")
            return backend(op, payload)
        ctrl = controller(serve, confirmations=2)
        self.addCleanup(ctrl.close)
        ctrl.start({"frame": 0}, "task")
        for step in (1, 2, 3):
            observe(ctrl, {"frame": step}, step)
        self.assertEqual(ctrl.replans, 0)
        self.assertEqual(ctrl.failure_streak, 1)

    def test_caption_failure_retries_with_latest_observation(self):
        backend = Backend(verdicts=[verdict("ongoing")] * 10)
        calls = []
        def serve(op, payload):
            if op == "caption":
                calls.append(payload)
                if len(calls) == 1:
                    raise TimeoutError("caption unavailable")
            return backend(op, payload)
        ctrl = controller(serve, caption_mode="instructor")
        self.addCleanup(ctrl.close)
        ctrl.start({"frame": 0}, "task")
        with self.assertRaises(TimeoutError):
            ctrl._caption_future.result(timeout=1)
        observe(ctrl, {"frame": 1}, 1)
        ctrl._caption_future.result(timeout=1)
        observe(ctrl, {"frame": 2}, 2)
        self.assertEqual(calls[1]["images"], [1] * 3)
        self.assertEqual(ctrl.caption, "fresh one")
        self.assertFalse(ctrl.stopped)

    def test_invalid_plan_retried_once_and_all_outputs_logged(self):
        calls = []
        def serve(op, payload):
            calls.append(payload)
            return {"phases": []} if len(calls) == 1 else {"phases": [phase("repair")]}
        ctrl = controller(serve)
        self.addCleanup(ctrl.close)
        ctrl.start({"frame": 0}, "task")
        self.assertEqual(len(calls), 2)
        self.assertIn("retry_feedback", calls[1])
        responses = [e for e in ctrl.events if e["event"] == "response"]
        self.assertEqual(responses[0]["response"]["phases"], [])
        self.assertEqual(ctrl.instruction, "repair")

    def test_connection_reset_retry_preserves_request_identity(self):
        requests = []
        def serve(op, payload):
            requests.append(payload["request_id"])
            if len(requests) == 1:
                raise ConnectionResetError("response lost after GPU generation")
            return {"phases": [phase("one")]}
        ctrl = controller(serve)
        self.addCleanup(ctrl.close)
        ctrl.start({"frame": 0}, "task")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], requests[1])

    def test_malformed_enum_values_are_model_output_errors(self):
        from hierarchical_controller import parse_plan, parse_verdict
        with self.assertRaises(ValueError):
            parse_verdict({"status": [], "confidence": .9, "evidence": "bad enum"})
        malformed = {**phase("one"), "instruction": []}
        with self.assertRaises(ValueError):
            parse_plan({"phases": [malformed]})

    def test_trace_survives_without_episode_summary_and_preserves_images(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = EventLog(Path(directory) / "episode", {"seed": 42})
            raw = b"image bytes"
            trace.append({"event": "request", "payload": {
                "images": [base64.b64encode(raw).decode()], "text": "full plan"}})
            events = [json.loads(x) for x in (trace.directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["payload"]["text"], "full plan")
            self.assertEqual((trace.directory / events[-1]["payload"]["images"][0]).read_bytes(), raw)
            self.assertTrue((trace.directory / "timeline.md").exists())

class ServiceAndReportTests(unittest.TestCase):
    def test_http_service_deduplicates_and_rejects_stale_requests(self):
        import ast
        import logging
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from hierarchical_controller import HTTPBackend
        source = Path(__file__).with_name("qwen_hierarchy_server.py")
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_handler")
        namespace = dict(threading=threading, time=time, logging=logging, json=json,
                         BaseHTTPRequestHandler=BaseHTTPRequestHandler,
                         decode_images=lambda data: data["images"], InvalidModelOutput=ValueError)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        class Model:
            calls = 0
            def structured(self, images, payload, operation):
                self.calls += 1
                return {"phases": [phase("one")], "raw_output": "complete"}
        model = Model()
        handler = namespace["make_handler"](model)
        handler.log_message = lambda *args: None
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = HTTPBackend(f"http://127.0.0.1:{server.server_port}")
            payload = {"images": ["x"] * 3, "session_id": "episode", "generation": 2,
                       "request_id": "request", "deadline_unix": time.time() + 60}
            first = backend("plan", payload)
            self.assertEqual(backend("plan", payload), first)
            self.assertEqual(model.calls, 1)
            with self.assertRaisesRegex(RuntimeError, "stale_or_expired"):
                backend("plan", {**payload, "generation": 1, "request_id": "old"})
            self.assertEqual(model.calls, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_comparison_excludes_failures_and_missing_timing(self):
        from hierarchy_report import compare
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a.json", Path(directory) / "b.json"
            common = {"seed": 42, "max_steps": 1200, "timing_protocol": "episode_start_before_reset_v1"}
            left.write_text(json.dumps({"task": {"episode_results": [
                {**common, "episode": 0, "success": True, "success_wall_seconds": 10},
                {**common, "episode": 1, "success": True, "success_wall_seconds": 10}]}}))
            right.write_text(json.dumps({"task": {"episode_results": [
                {**common, "episode": 0, "success": True, "success_wall_seconds": 15},
                {**common, "episode": 1, "success": False, "success_wall_seconds": None}]}}))
            report = compare(left, right)
            self.assertEqual(report["paired_episodes"], 2)
            self.assertEqual(report["paired_success_latency_count"], 1)
            self.assertEqual(report["mean_added_seconds"], 5)
            self.assertIsNone(report["episodes"][1]["delta_seconds"])


if __name__ == "__main__":
    unittest.main()
