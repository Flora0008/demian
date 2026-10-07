"""Run with python -m unittest discover -s openpi/scripts/eval -p 'test_hierarchical_controller.py'."""
import ast
import contextlib
import io
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

from hierarchical_controller import HierarchicalController, HierarchyConfig, parse_object, parse_plan, parse_verdict


def phase(name):
    return {"skill": "place", "instruction": name, "success_criteria": f"{name} visibly completed",
            "caption_type": "current_action", "caption": "caption for " + name}


def verdict(status="success", confidence=0.99):
    return {"status": status, "confidence": confidence, "evidence": "object released at target"}


class Backend:
    def __init__(self, phases=None, verdicts=None):
        self.phases = phases or [phase("one"), phase("two"), phase("three")]
        self.verdicts = list(verdicts or [])
        self.requests = []

    def __call__(self, operation, payload):
        self.requests.append((operation, payload))
        if operation == "plan":
            return {"phases": self.phases}
        if operation == "caption":
            return {"caption": "fresh " + payload["task_description"]}
        return self.verdicts.pop(0) if self.verdicts else verdict()


class SlowVerifierBackend(Backend):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, operation, payload):
        if operation == "verify":
            self.started.set()
            self.release.wait(timeout=2)
        return super().__call__(operation, payload)


def controller(backend=None, **kwargs):
    config_args = {"check_every": 1, "min_phase_steps": 1, "confirmations": 1,
                   "phase_timeout": 8, "async_verifier": False}
    config_args.update(kwargs)
    config = HierarchyConfig(**config_args)
    return HierarchicalController(backend or Backend(), config, encoder=lambda obs: [obs["frame"]] * 3)


def observe(ctrl, obs, step):
    return ctrl.observe(obs, step, at_chunk_boundary=True)


class ControllerTests(unittest.TestCase):
    def test_verifier_runs_in_background(self):
        backend = SlowVerifierBackend()
        ctrl = controller(backend, async_verifier=True)
        try:
            ctrl.start({"frame": 0}, "task")
            started = time.monotonic()
            self.assertFalse(observe(ctrl, {"frame": 1}, 1))
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.2)
            self.assertTrue(backend.started.wait(timeout=0.5))
            backend.release.set()
        finally:
            ctrl.close()

    def test_three_phases_and_fresh_instructor_per_phase(self):
        backend = Backend()
        ctrl = controller(backend, caption_mode="instructor")
        ctrl.start({"frame": 0}, "whole task")
        self.assertEqual(ctrl.caption, "fresh one")
        plan_request = next(p for op, p in backend.requests if op == "plan")
        self.assertEqual(plan_request["caption_mode"], "instructor")
        self.assertEqual(plan_request["max_phases"], 5)
        self.assertFalse(observe(ctrl, {"frame": 1}, 1))
        self.assertTrue(observe(ctrl, {"frame": 2}, 2))
        self.assertEqual((ctrl.instruction, ctrl.caption), ("two", "fresh two"))
        observe(ctrl, {"frame": 3}, 3)
        self.assertTrue(observe(ctrl, {"frame": 4}, 4))
        self.assertEqual(ctrl.caption, "fresh three")
        observe(ctrl, {"frame": 5}, 5)
        observe(ctrl, {"frame": 6}, 6)
        self.assertTrue(ctrl.finished)
        caps = [p for op, p in backend.requests if op == "caption"]
        self.assertEqual([p["images"][0] for p in caps], [0, 2, 4])
        self.assertEqual([p["task_description"] for p in caps], ["one", "two", "three"])

    def test_plan_is_limited_to_five_phases(self):
        with self.assertRaises(ValueError):
            parse_plan({"phases": [phase(str(i)) for i in range(6)]})

    def test_same_timestep_cannot_confirm_twice(self):
        ctrl = controller()
        ctrl.start({"frame": 0}, "task")
        observe(ctrl, {"frame": 1}, 1)
        observe(ctrl, {"frame": 1}, 1)
        self.assertEqual(ctrl.index, 0)

    def test_uncertain_resets_success_streak(self):
        backend = Backend(verdicts=[verdict(), verdict("uncertain"), verdict(), verdict()])
        ctrl = controller(backend, confirmations=2)
        ctrl.start({"frame": 0}, "task")
        for t in (1, 2, 3, 4):
            self.assertFalse(observe(ctrl, {"frame": t}, t))
        self.assertTrue(observe(ctrl, {"frame": 5}, 5))

    def test_low_confidence_cannot_advance(self):
        ctrl = controller(Backend(verdicts=[verdict(confidence=0.2)] * 3))
        ctrl.start({"frame": 0}, "task")
        for t in range(1, 4):
            observe(ctrl, {"frame": t}, t)
        self.assertEqual(ctrl.index, 0)

    def test_failure_replans_with_feedback_and_new_images(self):
        backend = Backend(verdicts=[verdict("failure")])
        ctrl = controller(backend)
        ctrl.start({"frame": 0}, "task")
        self.assertFalse(observe(ctrl, {"frame": 1}, 1))
        self.assertTrue(observe(ctrl, {"frame": 2}, 2))
        self.assertEqual(ctrl.replans, 1)
        plans = [p for op, p in backend.requests if op == "plan"]
        self.assertEqual(plans[-1]["images"], [2, 2, 2])
        self.assertIn("Failure", plans[-1]["feedback"])
        self.assertEqual(plans[-1]["history"][0]["outcome"], "not_confirmed")

    def test_timeout_is_not_success(self):
        ctrl = controller(Backend(verdicts=[verdict("ongoing")] * 8), max_replans=0)
        ctrl.start({"frame": 0}, "task")
        for t in range(1, 9):
            observe(ctrl, {"frame": t}, t)
        self.assertTrue(ctrl.stopped)
        self.assertFalse(ctrl.finished)
        self.assertIn("timeout", ctrl.error)

    def test_no_caption_ablation(self):
        ctrl = controller(caption_mode="none")
        ctrl.start({"frame": 0}, "task")
        self.assertIsNone(ctrl.caption)
        observe(ctrl, {"frame": 1}, 1)
        observe(ctrl, {"frame": 2}, 2)
        self.assertIsNone(ctrl.caption)

    def test_empty_instructor_caption_is_abstention(self):
        original = Backend()
        def backend(op, payload):
            return {"caption": ""} if op == "caption" else original(op, payload)
        ctrl = controller(backend, caption_mode="instructor")
        ctrl.start({"frame": 0}, "task")
        self.assertFalse(ctrl.stopped)
        self.assertIsNone(ctrl.caption)

    def test_unknown_skill_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_plan({"phases": [phase("one")]}, skills=["open"])

    def test_invalid_json_and_confidence_are_rejected(self):
        for text in ('{"phases": []} trailing', 'print("hello")', '[]'):
            with self.assertRaises(ValueError):
                parse_object(text)
        for conf in (float("nan"), True, "0.99", 2):
            with self.assertRaises(ValueError):
                parse_verdict(verdict(confidence=conf))
        with self.assertRaises(ValueError):
            parse_plan({"phases": []})

    def test_failed_service_stops_without_legacy_fallback(self):
        def backend(op, payload):
            raise TimeoutError("no service")
        ctrl = controller(backend)
        ctrl.start({"frame": 0}, "task")
        self.assertTrue(ctrl.stopped)
        self.assertIn("TimeoutError", ctrl.error)


class Array(list):
    def tolist(self):
        return list(self)


class FakeEnv:
    # Deliberately no `unwrapped` attribute: the planner has no simulator access.
    def __init__(self, success_at=None):
        self.t = 0
        self.success_at = success_at

    def reset(self, seed=None):
        self.t = 0
        return {"frame": 0}, {}

    def step(self, action):
        self.t += 1
        return {"frame": self.t}, 0.0, False, False, {"success": self.t == self.success_at}


class FakeClient:
    # Short chunks make the integration test reach a real chunk boundary on
    # each phase transition without pretending that production uses 2 steps.
    open_loop_horizon = 2

    def __init__(self):
        self.actions_from_chunk_completed = 0
        self.requests = []

    def act(self, obs, instruction, reasoning_text=None):
        # Mimic actual chunk cache: a changed prompt alone does not flush it.
        if self.actions_from_chunk_completed in (0, self.open_loop_horizon):
            self.requests.append((obs["frame"], instruction, reasoning_text))
            self.actions_from_chunk_completed = 0
        self.actions_from_chunk_completed += 1
        return {"raw_action": Array([0] * 7), "action_dict": {}}

    infer = act


class EvalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the real edited rollout function with fake env/client, without
        # importing GPU/simulator packages. This tests wiring, not visual quality.
        path = Path(__file__).with_name("run_robocasa_eval.py")
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_episode")
        namespace = {"RoboCasaClient": object, "AsyncInstructor": object,
                     "_subtitle_max_chars": lambda *args: 80,
                     "_wrap_subtitle_lines": lambda lines, *args: lines,
                     "np": SimpleNamespace(array2string=lambda *args, **kwargs: "[mock action]")}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        cls.run_episode = staticmethod(namespace["run_episode"])

    def rollout(self, success_at=None, hierarchy=True):
        ctrl = controller() if hierarchy else None
        client = FakeClient()
        if not hierarchy:
            client.open_loop_horizon = 8
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.run_episode(FakeEnv(success_at), client, "whole task", 14,
                                      True, record_video=False, hierarchy=ctrl)
        return result, client

    def test_phase_switch_flushes_real_rollout_chunk(self):
        result, client = self.rollout()
        self.assertEqual(client.requests, [(0, "one", "caption for one"),
                                          (2, "one", "caption for one"),
                                          (4, "two", "caption for two"),
                                          (6, "two", "caption for two"),
                                          (8, "three", "caption for three"),
                                          (10, "three", "caption for three")])
        self.assertEqual(result["steps"], 12)

    def test_verifier_success_cannot_forge_benchmark_success(self):
        result, _ = self.rollout()
        self.assertEqual(result["hierarchy"]["status"], "plan_finished")
        self.assertFalse(result["success"])

    def test_env_success_remains_authoritative(self):
        result, _ = self.rollout(success_at=3)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)

    def test_feature_off_keeps_task_only_loop(self):
        result, client = self.rollout(hierarchy=False)
        self.assertNotIn("hierarchy", result)
        self.assertEqual(client.requests, [(0, "whole task", None), (8, "whole task", None)])


if __name__ == "__main__":
    unittest.main()
