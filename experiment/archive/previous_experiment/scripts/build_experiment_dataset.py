"""Build the canonical Java tasks from CoderEval sources and the official run log."""

import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CoderEval" / "CoderEval4Java.json"
ARCHIVE = ROOT / "CoderEval" / "CoderEval-Input4Models.zip"
RAW_MEMBER = "CoderEval-Input4Models/CEJavaRaw.jsonl"
LOG = ROOT / "results" / "groundtruth_official.log"
DATASET = ROOT / "experiment" / "data" / "experiment_tasks.jsonl"
CSV = ROOT / "results" / "ground_truth_validation.csv"
REPORT = ROOT / "results" / "experiment_dataset_validation.json"

EXPECTED_FAIL = {
    "6367667f1a6d9265ec01745c": "addToClassPath",
    "636767791a6d9265ec018257": "addMessage",
    "636766801a6d9265ec017482": "readFrom",
}
EXPECTED_LEVELS = {
    "self_contained": 55,
    "slib_runnable": 25,
    "plib_runnable": 5,
    "class_runnable": 98,
    "file_runnable": 1,
    "project_runnable": 43,
}
TASK_LINE = re.compile(r"^([0-9a-f]{24}) ([01]) ([01]) (\d+)\.0$")


def main():
    ground = json.loads(SOURCE.read_text(encoding="utf-8"))["RECORDS"]
    ground_ids = [row["_id"] for row in ground]
    assert len(ground_ids) == len(set(ground_ids)) == 230

    with zipfile.ZipFile(ARCHIVE) as archive:
        raw = [json.loads(line) for line in archive.read(RAW_MEMBER).decode("utf-8").splitlines() if line.strip()]
    raw_ids = [row["question_id"] for row in raw]
    assert len(raw_ids) == len(set(raw_ids))
    raw_by_id = {row["question_id"]: row for row in raw}
    assert set(ground_ids) <= set(raw_by_id), "Raw input does not cover every ground truth task"

    results = {}
    cumulative = 0
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"[0-9a-f]{24}", fields[0]):
            continue
        match = TASK_LINE.fullmatch(line.strip())
        assert match, f"Unexpected task result format: {line!r}"
        task_id, ran, passed, total = match.groups()
        assert task_id not in results, f"Duplicate log task: {task_id}"
        assert ran == "1", f"Task did not run: {task_id}"
        cumulative += int(passed)
        assert int(total) == cumulative, f"Unexpected cumulative PASS count: {task_id}"
        results[task_id] = bool(int(passed))

    assert set(results) == set(ground_ids), "Log and CoderEval4Java.json task IDs differ"
    pass_ids = {task_id for task_id, passed in results.items() if passed}
    fail = {row["_id"]: row["name"] for row in ground if not results[row["_id"]]}
    assert len(pass_ids) == 227 and len(fail) == 3
    assert fail == EXPECTED_FAIL, f"FAIL tasks differ from expected comparison: {fail}"

    rows = []
    for source in ground:
        task_id = source["_id"]
        if task_id not in pass_ids:
            continue
        rows.append({
            "experiment_index": len(rows) + 1,
            "task_id": task_id,
            "project": source["project"],
            "class_name": source["class_name"],
            "file_name": source["file_name"],
            "package": source["package"],
            "method_name": source["name"],
            "human_label": source["human_label"],
            "dependency_level": source["level"],
            "signature": raw_by_id[task_id]["signature"],
            "ground_truth_code": source["code"],
            "lineno": source["lineno"],
            "end_lineno": source["end_lineno"],
        })

    DATASET.parent.mkdir(parents=True, exist_ok=True)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["task_id", "method_name", "dependency_level", "pass"])
        writer.writeheader()
        writer.writerows({
            "task_id": row["_id"], "method_name": row["name"],
            "dependency_level": row["level"], "pass": results[row["_id"]],
        } for row in ground)
    with DATASET.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    saved = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    saved_ids = [row["task_id"] for row in saved]
    levels = dict(Counter(row["dependency_level"] for row in saved))
    missing = {field: sum(not row.get(field) for row in saved)
               for field in ("signature", "human_label", "ground_truth_code", "dependency_level")}
    duplicate_count = len(saved_ids) - len(set(saved_ids))
    assert len(saved) == len(set(saved_ids)) == len(pass_ids) == 227
    assert duplicate_count == 0 and all(count == 0 for count in missing.values())
    assert set(saved_ids) == pass_ids == set(ground_ids) - set(fail)
    assert levels == EXPECTED_LEVELS, f"Unexpected dependency distribution: {levels}"

    report = {
        "raw_member": RAW_MEMBER,
        "join_key": "CoderEval4Java.json _id = CEJavaRaw.jsonl question_id",
        "original_task_count": len(ground),
        "raw_task_count": len(raw),
        "pass_count": len(pass_ids),
        "fail_count": len(fail),
        "fail_tasks": [{"task_id": task_id, "method_name": name} for task_id, name in fail.items()],
        "expected_fail_match": fail == EXPECTED_FAIL,
        "dataset_task_count": len(saved),
        "unique_task_id_count": len(set(saved_ids)),
        "duplicate_task_id_count": duplicate_count,
        "missing": missing,
        "dependency_distribution": levels,
        "expected_dependency_match": levels == EXPECTED_LEVELS,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
