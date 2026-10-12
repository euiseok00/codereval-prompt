"""Plan and run the canonical CoderEval prompt experiment."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "experiment" / "data" / "experiment_tasks.jsonl"
RESULTS = ROOT / "results" / "main_experiment"
PERSONA = "You are a highly skilled software developer who writes maintainable and robust code, carefully considering edge cases and exceptional conditions."
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
            "experiment_index": task["experiment_index"],
            "condition_id": condition_id,
            "persona": persona,
            "signature": signature,
            "cot": cot,
            "prompt": prompt_for(task, persona, signature, cot),
        }
        for task in tasks
        for condition_id, persona, signature, cot in CONDITIONS
    ]


def save_plan_file(path, content):
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing plan differs: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def plan():
    tasks = read_jsonl(DATASET)
    jobs = build_jobs(tasks)
    ids = [job["job_id"] for job in jobs]
    duplicates = len(ids) - len(set(ids))
    missing_prompt = sum(not job["prompt"] for job in jobs)
    assert len(tasks) == 227 and len(CONDITIONS) == 8 and len(jobs) == 1816
    assert duplicates == missing_prompt == 0
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    for task, group in zip(tasks, (jobs[i:i + 8] for i in range(0, len(jobs), 8))):
        for job, (condition_id, persona, signature, cot) in zip(group, CONDITIONS):
            assert (job["condition_id"], job["persona"], job["signature"], job["cot"]) == (condition_id, persona, signature, cot)
            assert job["prompt"] == prompt_for(task, persona, signature, cot)
            assert task["dependency_level"] not in job["prompt"]
    manifest = {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "tasks": len(tasks),
        "conditions": len(CONDITIONS),
        "jobs": len(jobs),
        "expected_jobs": len(jobs),
        "model": "gpt-5.6-sol",
        "api": "Responses",
        "reasoning_mode": "standard",
        "reasoning_effort": "none",
        "reasoning": {"mode": "standard", "effort": "none"},
        "condition_factors": [
            {"condition_id": name, "persona": p, "signature": s, "cot": c}
            for name, p, s, c in CONDITIONS
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_plan_file(RESULTS / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    save_plan_file(RESULTS / "job_plan.jsonl", "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs))
    print(f"tasks={len(tasks)} conditions={len(CONDITIONS)} jobs={len(jobs)} duplicate_job_id={duplicates} missing_prompt={missing_prompt}")
    print("C1-C8 factor mapping, prompt ON/OFF, dependency exclusion: PASS")
    for job in jobs:
        if job["task_id"] == "6367676d1a6d9265ec018229":
            print(f"\n{job['condition_id']} prompt:\n{job['prompt']}")


def planned_jobs():
    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_sha256"] != hashlib.sha256(DATASET.read_bytes()).hexdigest():
        raise RuntimeError("Dataset changed since plan; rerun plan and review results")
    return read_jsonl(RESULTS / "job_plan.jsonl")


def completed(path):
    rows = read_jsonl(path)
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate completed job in {path}")
    return {row["job_id"]: row for row in rows}


def append_result(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def value(obj, field):
    return getattr(obj, field, None) if obj is not None else None


def generate(confirmed, max_jobs=None):
    if not confirmed:
        raise SystemExit("generate requires --confirm-full-run")
    jobs = planned_jobs()
    done = completed(RESULTS / "generations.jsonl")
    from openai import OpenAI

    gateway_key = os.environ.get("BAZE_API_KEY")
    if not gateway_key:
        raise SystemExit("BAZE_API_KEY is required")
    client = OpenAI(api_key=gateway_key, base_url="https://factchat-cloud.mindlogic.ai/v1/gateway", max_retries=0)
    generated_count = 0
    with (RESULTS / "generations.jsonl").open("a", encoding="utf-8", newline="\n") as file:
        for job in jobs:
            if job["job_id"] in done:
                continue
            start = time.perf_counter()
            response = client.responses.create(
                model="gpt-5.6-sol",
                input=job["prompt"],
                reasoning={"mode": "standard", "effort": "none"},
                service_tier="default",
            )
            latency = time.perf_counter() - start
            usage = response.usage
            row = {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "condition_id": job["condition_id"],
                "raw_prompt": job["prompt"],
                "raw_response": response.model_dump(mode="json"),
                "generated_code": response.output_text,
                "request_model": "gpt-5.6-sol",
                "api_provider": "baze_gateway",
                "response_model": response.model,
                "response_id": response.id,
                "response_status": value(response, "status"),
                "input_tokens": value(usage, "input_tokens"),
                "cached_input_tokens": value(value(usage, "input_tokens_details"), "cached_tokens"),
                "cache_write_tokens": value(value(usage, "input_tokens_details"), "cache_write_tokens"),
                "output_tokens": value(usage, "output_tokens"),
                "reasoning_tokens": value(value(usage, "output_tokens_details"), "reasoning_tokens"),
                "total_tokens": value(usage, "total_tokens"),
                "raw_usage": usage.model_dump(mode="json") if usage is not None else None,
                "latency_seconds": latency,
                "retry_count": 0,
            }
            append_result(file, row)
            done[job["job_id"]] = row
            print(f"generated {job['job_id']}", flush=True)
            generated_count += 1
            if max_jobs is not None and generated_count >= max_jobs:
                break


EVALUATOR = r'''
import json, pathlib, subprocess, sys
item = json.load(sys.stdin)
root = pathlib.Path("/tmp/codereval-smoke")
root.mkdir(exist_ok=True)
source = root / "condition.jsonl"
output = pathlib.Path(str(source) + "_out.jsonl")
try:
    source.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    result = subprocess.run(
        ["python3", "/home/travis/builds/java/JavaExec.py", str(source), "1"],
        cwd="/home/travis/builds/java", text=True, capture_output=True, timeout=180
    )
    if result.returncode:
        raise RuntimeError(f"evaluator exit {result.returncode}: {result.stderr[-1000:]}")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1 or len(rows[0]["generate_results"]) != 1:
        raise RuntimeError(f"unexpected evaluator output: {len(rows)} rows")
    print(json.dumps(rows[0], ensure_ascii=False))
finally:
    source.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    root.rmdir()
'''


def evaluate():
    jobs = planned_jobs()
    generated = completed(RESULTS / "generations.jsonl")
    done = completed(RESULTS / "evaluations.jsonl")
    with (RESULTS / "evaluations.jsonl").open("a", encoding="utf-8", newline="\n") as file:
        for job in jobs:
            job_id = job["job_id"]
            if job_id in done or job_id not in generated:
                continue
            result = subprocess.run(
                ["docker", "exec", "-i", "codereval", "python3", "-c", EVALUATOR],
                input=json.dumps({"_id": job["task_id"], "generate_results": [generated[job_id]["generated_code"]]}, ensure_ascii=False),
                text=True, encoding="utf-8", capture_output=True, timeout=240,
            )
            if result.returncode:
                raise RuntimeError(f"Docker evaluator failed for {job_id}: {result.stderr[-1000:]}")
            raw = json.loads(result.stdout)
            outcome = raw["generate_results"][0]
            row = {
                "job_id": job_id,
                "task_id": job["task_id"],
                "condition_id": job["condition_id"],
                "verdict": "PASS" if outcome["is_pass"] else "FAIL",
                "compile_error": outcome["errormessage"] == "compile error",
                "test_exec_error": outcome["errormessage"] == "exec error!",
                "error_message": outcome["errormessage"],
                "error_detail_message": outcome["error_detail_message"],
                "raw_evaluation": raw,
            }
            append_result(file, row)
            done[job_id] = row
            print(f"evaluated {job_id}: {row['verdict']}", flush=True)


def summarize():
    jobs = planned_jobs()
    generated = completed(RESULTS / "generations.jsonl")
    evaluated = completed(RESULTS / "evaluations.jsonl")
    rows = []
    for condition_id, _, _, _ in CONDITIONS:
        ids = [job["job_id"] for job in jobs if job["condition_id"] == condition_id]
        grades = [evaluated[job_id]["verdict"] for job_id in ids if job_id in evaluated]
        rows.append({
            "condition_id": condition_id,
            "jobs": len(ids),
            "generated": sum(job_id in generated for job_id in ids),
            "evaluated": len(grades),
            "pass": grades.count("PASS"),
            "fail": grades.count("FAIL"),
        })
    with (RESULTS / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "jobs": len(jobs),
        "generated": len(generated),
        "evaluated": len(evaluated),
        "pass": sum(row["pass"] for row in rows),
        "fail": sum(row["fail"] for row in rows),
        "conditions": rows,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("plan")
    generate_parser = modes.add_parser("generate")
    generate_parser.add_argument("--confirm-full-run", action="store_true")
    generate_parser.add_argument("--max-jobs", type=int, choices=range(1, 1817))
    modes.add_parser("evaluate")
    modes.add_parser("summarize")
    args = parser.parse_args()
    if args.mode == "plan":
        plan()
    elif args.mode == "generate":
        generate(args.confirm_full_run, args.max_jobs)
    elif args.mode == "evaluate":
        evaluate()
    else:
        summarize()


if __name__ == "__main__":
    main()
