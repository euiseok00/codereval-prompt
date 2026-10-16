"""Plan, generate, and evaluate the main CoderEval Java prompt experiment."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "experiment" / "data" / "CEJavaPrompt.jsonl"
JOBS = ROOT / "experiment" / "data" / "CEJavaPromptJobs.jsonl"
RESULTS = ROOT / "results" / "main"
CONFIG = RESULTS / "ExperimentConfig.json"

MODEL = "gpt-5.6-sol"
TEMPERATURE = 0.6
RUNS = (1, 2, 3)
PERSONA = (
    "You are a highly skilled software developer who writes maintainable and robust code, "
    "carefully considering edge cases and exceptional conditions."
)
COT = "Let's think step by step."
OUTPUT = "Return only the final Java method code."
CONDITIONS = (
    ("C1", False, False, False),
    ("C2", False, True, False),
    ("C3", False, False, True),
    ("C4", False, True, True),
    ("C5", True, False, False),
    ("C6", True, True, False),
    ("C7", True, False, True),
    ("C8", True, True, True),
)


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def save_immutable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing file differs: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def prompt_for(task, persona, signature, cot):
    parts = []
    if persona:
        parts.append(PERSONA)
    parts.extend((task["human_label"], task["method_name"]))
    if signature:
        parts.append(task["signature"])
    if cot:
        parts.append(COT)
    parts.append(OUTPUT)
    return "\n\n".join(parts)


def build_jobs(tasks):
    return [
        {
            "job_id": f"{task['task_id']}_{condition_id}",
            "task_id": task["task_id"],
            "condition_id": condition_id,
            "persona": persona,
            "signature": signature,
            "cot": cot,
            "prompt": prompt_for(task, persona, signature, cot),
        }
        for task in tasks
        for condition_id, persona, signature, cot in CONDITIONS
    ]


def plan():
    tasks = read_jsonl(DATASET)
    if not tasks:
        raise SystemExit("CEJavaPrompt.jsonl not found. Run BuildCEJavaPrompt.py first.")
    jobs = build_jobs(tasks)
    ids = [job["job_id"] for job in jobs]
    assert len(tasks) == 227
    assert len(jobs) == len(set(ids)) == 1816

    job_content = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    save_immutable(JOBS, job_content)

    config = {
        "dataset": "experiment/data/CEJavaPrompt.jsonl",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "job_plan": "experiment/data/CEJavaPromptJobs.jsonl",
        "tasks": 227,
        "conditions": 8,
        "jobs_per_run": 1816,
        "runs": 3,
        "total_generations": 5448,
        "model": MODEL,
        "reasoning_effort": "none",
        "temperature": TEMPERATURE,
        "metrics": ["Pass@1", "Pass@3"],
        "condition_factors": [
            {
                "condition_id": condition_id,
                "persona": persona,
                "signature": signature,
                "cot": cot,
            }
            for condition_id, persona, signature, cot in CONDITIONS
        ],
    }
    save_immutable(CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print("tasks=227 conditions=8 jobs_per_run=1816 runs=3 total_generations=5448")


def planned_jobs():
    if not CONFIG.exists() or not JOBS.exists():
        raise SystemExit("Experiment plan not found. Run the plan command first.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["dataset_sha256"] != hashlib.sha256(DATASET.read_bytes()).hexdigest():
        raise RuntimeError("CEJavaPrompt.jsonl changed after planning")
    jobs = read_jsonl(JOBS)
    assert len(jobs) == 1816
    return jobs


def completed(path):
    rows = read_jsonl(path)
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate job_id in {path}")
    return {row["job_id"]: row for row in rows}


def value(obj, field):
    return getattr(obj, field, None) if obj is not None else None


def iso_created_at(response):
    created = value(response, "created_at")
    if isinstance(created, (int, float)):
        return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
    if created is not None:
        return str(created)
    return datetime.now(timezone.utc).isoformat()


def request_response(client, prompt):
    last_error = None
    for retry_count in range(3):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                reasoning={"effort": "none"},
                temperature=TEMPERATURE,
            )
            return response, retry_count
        except Exception as exc:  # API failures are not code-evaluation failures.
            last_error = exc
            if retry_count == 2:
                break
            time.sleep(2 ** retry_count)
    raise RuntimeError("API request failed after 3 attempts") from last_error


def generation_path(run):
    return RESULTS / "GenerateResultJava" / f"Run{run}.jsonl"


def evaluation_path(run):
    return RESULTS / "EvaluateResultJava" / f"Run{run}.jsonl"


def generate(run, confirmed, max_jobs=None):
    if run not in RUNS:
        raise ValueError(run)
    if not confirmed:
        raise SystemExit("generate requires --confirm-full-run")

    jobs = planned_jobs()
    path = generation_path(run)
    done = completed(path)

    from openai import OpenAI

    client = OpenAI(max_retries=0)
    generated_count = 0
    for job in jobs:
        if job["job_id"] in done:
            continue
        response, retry_count = request_response(client, job["prompt"])
        usage = response.usage
        row = {
            "job_id": job["job_id"],
            "task_id": job["task_id"],
            "condition_id": job["condition_id"],
            "generated_code": response.output_text,
            "response_id": response.id,
            "response_model": response.model,
            "created_at": iso_created_at(response),
            "input_tokens": value(usage, "input_tokens"),
            "cached_input_tokens": value(value(usage, "input_tokens_details"), "cached_tokens"),
            "output_tokens": value(usage, "output_tokens"),
            "reasoning_tokens": value(value(usage, "output_tokens_details"), "reasoning_tokens"),
            "total_tokens": value(usage, "total_tokens"),
            "retry_count": retry_count,
        }
        append_jsonl(path, row)
        done[job["job_id"]] = row
        generated_count += 1
        print(f"Run{run} generated {job['job_id']}", flush=True)
        if max_jobs is not None and generated_count >= max_jobs:
            break


EVALUATOR = r'''
import json, pathlib, subprocess, sys
item = json.load(sys.stdin)
root = pathlib.Path("/tmp/codereval-prompt-main")
root.mkdir(exist_ok=True)
source = root / "condition.jsonl"
output = pathlib.Path(str(source) + "_out.jsonl")
try:
    source.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            ["python3", "/home/travis/builds/java/JavaExec.py", str(source), "1"],
            cwd="/home/travis/builds/java", text=True, capture_output=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"timeout": True}))
        raise SystemExit(0)
    if result.returncode:
        raise RuntimeError(f"evaluator exit {result.returncode}: {result.stderr[-1000:]}")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1 or len(rows[0]["generate_results"]) != 1:
        raise RuntimeError(f"unexpected evaluator output: {len(rows)} rows")
    print(json.dumps(rows[0], ensure_ascii=False))
finally:
    source.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    try:
        root.rmdir()
    except OSError:
        pass
'''


def classify_failure(outcome):
    message = outcome.get("errormessage", "")
    if message == "compile error":
        return "COMPILE_ERROR"
    if message == "exec error!":
        return "RUNTIME_ERROR"
    return "TEST_FAILURE"


def evaluate(run):
    if run not in RUNS:
        raise ValueError(run)
    jobs = planned_jobs()
    generated = completed(generation_path(run))
    path = evaluation_path(run)
    done = completed(path)

    for job in jobs:
        job_id = job["job_id"]
        if job_id in done or job_id not in generated:
            continue
        code = generated[job_id].get("generated_code")
        if not isinstance(code, str) or not code.strip():
            row = {
                "job_id": job_id,
                "task_id": job["task_id"],
                "condition_id": job["condition_id"],
                "outcome": "FAIL",
                "failure_type": "INVALID_OUTPUT",
                "error_message": "empty generated output",
                "error_detail_message": "",
            }
            append_jsonl(path, row)
            done[job_id] = row
            continue

        try:
            result = subprocess.run(
                ["docker", "exec", "-i", "codereval", "python3", "-c", EVALUATOR],
                input=json.dumps(
                    {"_id": job["task_id"], "generate_results": [code]},
                    ensure_ascii=False,
                ),
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            row = {
                "job_id": job_id,
                "task_id": job["task_id"],
                "condition_id": job["condition_id"],
                "outcome": "FAIL",
                "failure_type": "TIMEOUT",
                "error_message": "Docker evaluation timeout",
                "error_detail_message": "",
            }
            append_jsonl(path, row)
            done[job_id] = row
            continue

        if result.returncode:
            raise RuntimeError(f"Docker evaluator failed for {job_id}: {result.stderr[-1000:]}")
        raw = json.loads(result.stdout)
        if raw.get("timeout"):
            verdict = "FAIL"
            failure_type = "TIMEOUT"
            error_message = "JavaExec timeout"
            error_detail = ""
        else:
            outcome = raw["generate_results"][0]
            verdict = "PASS" if outcome["is_pass"] else "FAIL"
            failure_type = None if verdict == "PASS" else classify_failure(outcome)
            error_message = outcome.get("errormessage", "")
            error_detail = outcome.get("error_detail_message", "")

        row = {
            "job_id": job_id,
            "task_id": job["task_id"],
            "condition_id": job["condition_id"],
            "outcome": verdict,
            "failure_type": failure_type,
            "error_message": error_message,
            "error_detail_message": error_detail,
        }
        append_jsonl(path, row)
        done[job_id] = row
        print(f"Run{run} evaluated {job_id}: {verdict}", flush=True)


def status():
    print(f"jobs={len(read_jsonl(JOBS))}")
    for run in RUNS:
        print(
            f"Run{run}: generated={len(read_jsonl(generation_path(run)))} "
            f"evaluated={len(read_jsonl(evaluation_path(run)))}"
        )


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("plan")

    generate_parser = modes.add_parser("generate")
    generate_parser.add_argument("--run", type=int, choices=RUNS, required=True)
    generate_parser.add_argument("--confirm-full-run", action="store_true")
    generate_parser.add_argument("--max-jobs", type=int, choices=range(1, 1817))

    evaluate_parser = modes.add_parser("evaluate")
    evaluate_parser.add_argument("--run", type=int, choices=RUNS, required=True)
    modes.add_parser("status")

    args = parser.parse_args()
    if args.mode == "plan":
        plan()
    elif args.mode == "generate":
        generate(args.run, args.confirm_full_run, args.max_jobs)
    elif args.mode == "evaluate":
        evaluate(args.run)
    else:
        status()


if __name__ == "__main__":
    main()
