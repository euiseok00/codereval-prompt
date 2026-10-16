"""Offline checks for the main CoderEval Java prompt experiment."""

import unittest

import AnalyzeCEJavaPrompt as analysis
import RunCEJavaPrompt as runner


class MainExperimentTests(unittest.TestCase):
    def test_eight_conditions_create_1816_unique_jobs(self):
        tasks = [
            {
                "task_id": f"{i:024x}",
                "method_name": "m",
                "human_label": "Do something.",
                "signature": "public void m(){",
                "dependency_level": "self_contained",
            }
            for i in range(227)
        ]
        jobs = runner.build_jobs(tasks)
        self.assertEqual(len(jobs), 1816)
        self.assertEqual(len({job["job_id"] for job in jobs}), 1816)

    def test_prompt_factor_mapping(self):
        task = {
            "task_id": "0" * 24,
            "method_name": "m",
            "human_label": "Do something.",
            "signature": "public void m(){",
            "dependency_level": "self_contained",
        }
        for condition_id, persona, signature, cot in runner.CONDITIONS:
            prompt = runner.prompt_for(task, persona, signature, cot)
            self.assertEqual(runner.PERSONA in prompt, persona, condition_id)
            self.assertEqual(task["signature"] in prompt, signature, condition_id)
            self.assertEqual(runner.COT in prompt, cot, condition_id)
            self.assertNotIn(task["dependency_level"], prompt)
            self.assertTrue(prompt.endswith(runner.OUTPUT))

    def test_pass_metrics(self):
        self.assertEqual(analysis.pass_metrics(["FAIL", "FAIL", "FAIL"]), (0, 0.0, 0))
        self.assertEqual(analysis.pass_metrics(["PASS", "FAIL", "FAIL"]), (1, 1 / 3, 1))
        self.assertEqual(analysis.pass_metrics(["PASS", "PASS", "FAIL"]), (2, 2 / 3, 1))
        self.assertEqual(analysis.pass_metrics(["PASS", "PASS", "PASS"]), (3, 1.0, 1))

    def test_failure_classification(self):
        self.assertEqual(runner.classify_failure({"errormessage": "compile error"}), "COMPILE_ERROR")
        self.assertEqual(runner.classify_failure({"errormessage": "exec error!"}), "RUNTIME_ERROR")
        self.assertEqual(runner.classify_failure({"errormessage": ""}), "TEST_FAILURE")


if __name__ == "__main__":
    unittest.main()
