"""Compute Pass@1 and Pass@3 for the three-run Java prompt experiment."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from RunCEJavaPrompt import CONDITIONS, JOBS, RESULTS, ROOT, RUNS, read_jsonl


DATASET = ROOT / "experiment" / "data" / "CEJavaPrompt.jsonl"
OUTPUT = RESULTS / "AnalysisJava"


def indexed(rows, name):
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate job_id in {name}")
    return {row["job_id"]: row for row in rows}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def pass_metrics(outcomes):
    if len(outcomes) != 3:
        raise ValueError("Pass@1/Pass@3 require exactly three independent outcomes")
    pass_count = sum(outcome == "PASS" for outcome in outcomes)
    return pass_count, pass_count / 3, int(pass_count > 0)


def aggregate(rows, dimensions):
    count = len(rows)
    return {
        **dimensions,
        "job_count": count,
        "pass_attempt_count": sum(row["pass_count"] for row in rows),
        "pass_at_1": sum(row["pass_at_1"] for row in rows) / count,
        "pass_at_3_success_count": sum(row["pass_at_3"] for row in rows),
        "pass_at_3": sum(row["pass_at_3"] for row in rows) / count,
    }


def main():
    tasks = read_jsonl(DATASET)
    jobs = read_jsonl(JOBS)
    assert len(tasks) == 227 and len(jobs) == 1816
    task_by_id = {task["task_id"]: task for task in tasks}
    assert len(task_by_id) == 227

    evaluations = {}
    for run in RUNS:
        path = RESULTS / "EvaluateResultJava" / f"Run{run}.jsonl"
        rows = read_jsonl(path)
        assert len(rows) == 1816, f"Run{run} evaluations: expected 1816, got {len(rows)}"
        evaluations[run] = indexed(rows, f"Run{run}")

    condition_factors = {
        condition_id: (persona, signature, cot)
        for condition_id, persona, signature, cot in CONDITIONS
    }

    per_job = []
    failure_counts = Counter()
    for job in jobs:
        job_id = job["job_id"]
        outcomes = [evaluations[run][job_id]["outcome"] for run in RUNS]
        pass_count, pass_at_1, pass_at_3 = pass_metrics(outcomes)
        for run in RUNS:
            row = evaluations[run][job_id]
            if row["outcome"] == "FAIL" and row.get("failure_type"):
                failure_counts[row["failure_type"]] += 1
        per_job.append({
            "job_id": job_id,
            "task_id": job["task_id"],
            "condition_id": job["condition_id"],
            "dependency_level": task_by_id[job["task_id"]]["dependency_level"],
            "pass_count": pass_count,
            "pass_at_1": pass_at_1,
            "pass_at_3": pass_at_3,
        })

    by_condition = defaultdict(list)
    by_dependency = defaultdict(list)
    by_cell = defaultdict(list)
    for row in per_job:
        by_condition[row["condition_id"]].append(row)
        by_dependency[row["dependency_level"]].append(row)
        by_cell[row["dependency_level"], row["condition_id"]].append(row)

    condition_rows = []
    for condition_id, persona, signature, cot in CONDITIONS:
        condition_rows.append(aggregate(by_condition[condition_id], {
            "condition_id": condition_id,
            "persona_on": persona,
            "signature_on": signature,
            "cot_on": cot,
        }))

    dependency_rows = [
        aggregate(rows, {"dependency_level": dependency})
        for dependency, rows in sorted(by_dependency.items())
    ]

    dependency_counts = Counter(task["dependency_level"] for task in tasks)
    cell_rows = []
    for dependency in dependency_counts:
        for condition_id, persona, signature, cot in CONDITIONS:
            rows = by_cell[dependency, condition_id]
            assert len(rows) == dependency_counts[dependency]
            cell_rows.append(aggregate(rows, {
                "dependency_level": dependency,
                "condition_id": condition_id,
                "persona_on": persona,
                "signature_on": signature,
                "cot_on": cot,
            }))

    overall = aggregate(per_job, {})
    summary = {
        "tasks": 227,
        "conditions": 8,
        "runs": 3,
        "jobs": 1816,
        "evaluations": 5448,
        "pass_at_1": overall["pass_at_1"],
        "pass_at_3": overall["pass_at_3"],
        "failure_type_counts": dict(sorted(failure_counts.items())),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "PassAtKJava.csv", per_job)
    write_csv(OUTPUT / "ConditionResultJava.csv", condition_rows)
    write_csv(OUTPUT / "DependencyResultJava.csv", dependency_rows)
    write_csv(OUTPUT / "DependencyConditionJava.csv", cell_rows)
    (OUTPUT / "SummaryJava.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
