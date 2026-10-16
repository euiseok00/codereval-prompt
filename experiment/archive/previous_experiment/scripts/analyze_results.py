"""Summarize correctness and token usage for the completed main experiment."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from run_experiment import CONDITIONS, ROOT, read_jsonl


MAIN = ROOT / "results" / "main_experiment"
OUTPUT = ROOT / "results" / "analysis"
DEPENDENCIES = (
    "self_contained",
    "slib_runnable",
    "plib_runnable",
    "class_runnable",
    "file_runnable",
    "project_runnable",
)
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def indexed(rows, name):
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate job_id in {name}")
    return {row["job_id"]: row for row in rows}


def summarize(rows, dimensions):
    count = len(rows)
    passed = sum(row["verdict"] == "PASS" for row in rows)
    result = {**dimensions, "task_count": count, "pass_count": passed,
              "fail_count": count - passed, "accuracy": passed / count}
    for field in TOKEN_FIELDS:
        values = [row[field] for row in rows]
        result[f"mean_{field}"] = mean(values)
        result[f"median_{field}"] = median(values)
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    tasks = read_jsonl(ROOT / "experiment" / "data" / "experiment_tasks.jsonl")
    plans = read_jsonl(MAIN / "job_plan.jsonl")
    generation_rows = read_jsonl(MAIN / "generations.jsonl")
    evaluation_rows = read_jsonl(MAIN / "evaluations.jsonl")
    task_by_id = {task["task_id"]: task for task in tasks}
    assert len(tasks) == len(task_by_id) == 227
    assert set(task["dependency_level"] for task in tasks) == set(DEPENDENCIES)
    assert len(plans) == len(generation_rows) == len(evaluation_rows) == 1816

    plan_by_id = indexed(plans, "job_plan")
    generated = indexed(generation_rows, "generations")
    evaluated = indexed(evaluation_rows, "evaluations")
    assert len(plan_by_id) == 1816
    missing_generations = set(plan_by_id) - set(generated)
    missing_evaluations = set(plan_by_id) - set(evaluated)
    assert not missing_generations and not missing_evaluations
    assert set(plan_by_id) == set(generated) == set(evaluated)

    factors = {condition_id: (persona, signature, cot)
               for condition_id, persona, signature, cot in CONDITIONS}
    assert len(factors) == 8
    joined = []
    providers = defaultdict(list)
    for job_id, plan in plan_by_id.items():
        generation = generated[job_id]
        evaluation = evaluated[job_id]
        task_id = plan["task_id"]
        condition_id = plan["condition_id"]
        assert task_id in task_by_id and condition_id in factors
        assert (plan["persona"], plan["signature"], plan["cot"]) == factors[condition_id]
        assert all(row["task_id"] == task_id and row["condition_id"] == condition_id
                   for row in (generation, evaluation))
        assert generation["raw_prompt"] == plan["prompt"]
        assert generation["response_model"] == "gpt-5.6-sol"
        assert generation["reasoning_tokens"] == 0
        assert all(isinstance(generation.get(field), int) and generation[field] >= 0
                   for field in TOKEN_FIELDS)
        assert evaluation["verdict"] in ("PASS", "FAIL")
        dependency = task_by_id[task_id]["dependency_level"]
        # Rows saved before the gateway switch have no provider field.
        provider = generation.get("api_provider", "openai_direct")
        assert provider in ("openai_direct", "baze_gateway")
        providers[provider].append(generation)
        joined.append({"dependency_level": dependency, "condition_id": condition_id,
                       "verdict": evaluation["verdict"],
                       **{field: generation[field] for field in TOKEN_FIELDS}})

    by_cell = defaultdict(list)
    by_condition = defaultdict(list)
    by_dependency = defaultdict(list)
    for row in joined:
        by_cell[row["dependency_level"], row["condition_id"]].append(row)
        by_condition[row["condition_id"]].append(row)
        by_dependency[row["dependency_level"]].append(row)

    dependency_counts = Counter(task["dependency_level"] for task in tasks)
    cell_rows = []
    for dependency in DEPENDENCIES:
        for condition_id, persona, signature, cot in CONDITIONS:
            group = by_cell[dependency, condition_id]
            assert len(group) == dependency_counts[dependency]
            cell_rows.append(summarize(group, {
                "dependency_level": dependency, "condition_id": condition_id,
                "persona_on": persona, "signature_on": signature, "cot_on": cot,
            }))
    condition_rows = [summarize(by_condition[condition_id], {
        "condition_id": condition_id, "persona_on": persona,
        "signature_on": signature, "cot_on": cot,
    }) for condition_id, persona, signature, cot in CONDITIONS]
    dependency_rows = [summarize(by_dependency[dependency], {
        "dependency_level": dependency,
    }) for dependency in DEPENDENCIES]
    assert len(cell_rows) == 48 and len(condition_rows) == 8 and len(dependency_rows) == 6
    assert sum(row["pass_count"] for row in condition_rows) == 671
    assert sum(row["fail_count"] for row in condition_rows) == 1145

    integrity = {
        "planned_jobs": len(plans),
        "unique_job_ids": len(plan_by_id),
        "generations": len(generation_rows),
        "evaluations": len(evaluation_rows),
        "missing_generations": len(missing_generations),
        "missing_evaluations": len(missing_evaluations),
        "duplicate_job_ids": 0,
        "response_model_all_gpt_5_6_sol": True,
        "reasoning_tokens_all_zero": True,
        "missing_token_counts": {field: sum(row.get(field) is None for row in generation_rows)
                                 for field in ("input_tokens", "output_tokens", "total_tokens")},
        "providers": {provider: {
            "count": len(rows),
            "response_models": sorted({row["response_model"] for row in rows}),
            "reasoning_tokens_all_zero": all(row["reasoning_tokens"] == 0 for row in rows),
        } for provider, rows in sorted(providers.items())},
    }
    assert set(integrity["providers"]) == {"openai_direct", "baze_gateway"}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = (
        OUTPUT / "dependency_condition.csv",
        OUTPUT / "condition_overall.csv",
        OUTPUT / "dependency_overall.csv",
        OUTPUT / "data_integrity.json",
    )
    write_csv(paths[0], cell_rows)
    write_csv(paths[1], condition_rows)
    write_csv(paths[2], dependency_rows)
    paths[3].write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Integrity: 1816 planned, 1816 unique, 0 missing generations/evaluations, 0 duplicates; model and reasoning: PASS; token fields missing: 0")
    print("\nAccuracy by dependency × condition")
    print("dependency_level " + " ".join(f"{condition_id:>6}" for condition_id in factors))
    for dependency in DEPENDENCIES:
        print(f"{dependency:<18}" + " ".join(f"{row['accuracy']:>6.3f}" for row in cell_rows
                                                     if row["dependency_level"] == dependency))
    print("\nCondition overall: accuracy, mean total tokens, median total tokens")
    for row in condition_rows:
        print(f"{row['condition_id']}: {row['accuracy']:.3f}, {row['mean_total_tokens']:.2f}, {row['median_total_tokens']}")
    print("\nDependency overall: accuracy")
    for row in dependency_rows:
        print(f"{row['dependency_level']}: {row['accuracy']:.3f}")
    print("\nProvider counts: " + ", ".join(f"{provider}={data['count']}"
                                           for provider, data in integrity["providers"].items()))
    print("Files: " + ", ".join(str(path.relative_to(ROOT)).replace("\\", "/") for path in paths))


if __name__ == "__main__":
    main()
