"""Offline checks for the experiment runner."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_experiment as runner


class ExperimentRunnerTests(unittest.TestCase):
    def test_plan_has_unique_jobs_and_exact_factors(self):
        tasks = runner.read_jsonl(runner.DATASET)
        jobs = runner.build_jobs(tasks)
        self.assertEqual((len(tasks), len(jobs), len({job["job_id"] for job in jobs})), (227, 1816, 1816))
        task = next(task for task in tasks if task["task_id"] == "6367676d1a6d9265ec018229")
        for job, (_, persona, signature, cot) in zip(jobs[:8], runner.CONDITIONS):
            self.assertEqual(job["prompt"], runner.prompt_for(task, persona, signature, cot))
            self.assertEqual(runner.PERSONA in job["prompt"], persona)
            self.assertEqual(task["signature"] in job["prompt"], signature)
            self.assertEqual(runner.COT in job["prompt"], cot)
            self.assertNotIn(task["dependency_level"], job["prompt"])

    def test_generate_needs_confirmation_before_any_request(self):
        with self.assertRaisesRegex(SystemExit, "--confirm-full-run"):
            runner.generate(False)

    def test_generate_records_raw_response_and_resumes_without_api(self):
        job = {"job_id": "task_C1", "task_id": "task", "condition_id": "C1", "prompt": "raw prompt"}
        raw_usage = {"input_tokens": 20, "output_tokens": 17, "total_tokens": 37}
        usage = SimpleNamespace(
            input_tokens=20, input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens=17, output_tokens_details=SimpleNamespace(reasoning_tokens=5),
            total_tokens=37, model_dump=lambda **_: raw_usage,
        )
        response = SimpleNamespace(
            usage=usage, output_text="```java\ncode\n```", model="gpt-5.6-sol", id="response-id",
            status="completed", model_dump=lambda **_: {"output": "raw response"},
        )
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
        factory = Mock(return_value=client)
        fake_openai = SimpleNamespace(OpenAI=factory)
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as directory:
            with patch.object(runner, "RESULTS", Path(directory)), patch.object(runner, "planned_jobs", return_value=[job]), patch.dict(sys.modules, {"openai": fake_openai}), patch.dict(os.environ, {"BAZE_API_KEY": "test-only"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.generate(True)
                    runner.generate(True)
                factory.assert_called_with(api_key="test-only", base_url="https://factchat-cloud.mindlogic.ai/v1/gateway", max_retries=0)
                self.assertEqual(client.responses.create.call_count, 1)
                request = client.responses.create.call_args.kwargs
                self.assertEqual(request["input"], "raw prompt")
                self.assertEqual(request["model"], "gpt-5.6-sol")
                self.assertEqual(request["reasoning"], {"mode": "standard", "effort": "none"})
                for omitted in ("previous_response_id", "conversation", "temperature", "top_p", "top_logprobs", "max_output_tokens"):
                    self.assertNotIn(omitted, request)
                saved = runner.read_jsonl(Path(directory) / "generations.jsonl")
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["job_id"], job["job_id"])
                self.assertEqual(saved[0]["raw_prompt"], job["prompt"])
                self.assertEqual(saved[0]["generated_code"], response.output_text)
                self.assertEqual(saved[0]["raw_response"], {"output": "raw response"})
                self.assertEqual(saved[0]["api_provider"], "baze_gateway")
                self.assertEqual((saved[0]["input_tokens"], saved[0]["output_tokens"], saved[0]["reasoning_tokens"], saved[0]["total_tokens"]), (20, 17, 5, 37))
                self.assertEqual(saved[0]["raw_usage"], raw_usage)
                self.assertEqual((saved[0]["response_id"], saved[0]["response_model"], saved[0]["response_status"]), ("response-id", "gpt-5.6-sol", "completed"))
                self.assertGreaterEqual(saved[0]["latency_seconds"], 0)
                self.assertEqual(saved[0]["retry_count"], 0)
                self.assertIsNone(saved[0]["cache_write_tokens"])

    def test_generate_max_jobs_stops_after_first_pending_job(self):
        jobs = [
            {"job_id": "task_C1", "task_id": "task", "condition_id": "C1", "prompt": "first"},
            {"job_id": "task_C2", "task_id": "task", "condition_id": "C2", "prompt": "second"},
        ]
        response = SimpleNamespace(
            usage=None, output_text="code", model="gpt-5.6-sol", id="response-id",
            status="completed", model_dump=lambda **_: {"output": "raw response"},
        )
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as directory:
            with patch.object(runner, "RESULTS", Path(directory)), patch.object(runner, "planned_jobs", return_value=jobs), patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **_: client)}), patch.dict(os.environ, {"BAZE_API_KEY": "test-only"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.generate(True, 1)
                self.assertEqual(client.responses.create.call_count, 1)
                self.assertEqual([row["job_id"] for row in runner.read_jsonl(Path(directory) / "generations.jsonl")], ["task_C1"])

    def test_evaluate_preserves_code_and_resumes(self):
        code = "```java\npublic void x() {}\n```"
        job = {"job_id": "task_C1", "task_id": "task", "condition_id": "C1"}
        outcome = {"generate_results": [{"is_pass": False, "errormessage": "compile error", "error_detail_message": "bad signature"}]}
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as directory:
            results = Path(directory)
            (results / "generations.jsonl").write_text(json.dumps({**job, "generated_code": code}) + "\n", encoding="utf-8")
            with patch.object(runner, "RESULTS", results), patch.object(runner, "planned_jobs", return_value=[job]), patch.object(runner.subprocess, "run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = json.dumps(outcome)
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.evaluate()
                    runner.evaluate()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(json.loads(run.call_args.kwargs["input"])["generate_results"], [code])
                saved = runner.read_jsonl(results / "evaluations.jsonl")
                self.assertEqual((len(saved), saved[0]["verdict"], saved[0]["compile_error"]), (1, "FAIL", True))


if __name__ == "__main__":
    unittest.main()
