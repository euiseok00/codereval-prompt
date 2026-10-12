# Canonical Java experiment dataset

Run `python experiment/scripts/build_experiment_dataset.py` from this repository to rebuild `experiment/data/experiment_tasks.jsonl`, `results/ground_truth_validation.csv`, and `results/experiment_dataset_validation.json`. The script reads the preserved official run log at `results/groundtruth_official.log`; it does not run GroundTruth.py.

The ground truth source is `CoderEval/CoderEval4Java.json`, whose `RECORDS` array has 230 tasks. The signature source is `CoderEval/CoderEval-Input4Models.zip`, member `CoderEval-Input4Models/CEJavaRaw.jsonl`, whose JSONL rows have `question_id`, `input`, `signature`, and `docstring`. The Raw file has 239 unique IDs. All 230 ground truth IDs match exactly one Raw row using `RECORDS._id = question_id`; the nine additional Raw rows are outside the ground truth source. Method names are not used for joining.

| Canonical field | Source field |
| --- | --- |
| `experiment_index` | Sequential, one-based index in original `RECORDS` order after PASS filtering |
| `task_id` | `CoderEval4Java.json` → `RECORDS._id` |
| `project` | `RECORDS.project` |
| `class_name` | `RECORDS.class_name` |
| `file_name` | `RECORDS.file_name` |
| `package` | `RECORDS.package` |
| `method_name` | `RECORDS.name` |
| `human_label` | `RECORDS.human_label` |
| `dependency_level` | `RECORDS.level` |
| `signature` | `CEJavaRaw.jsonl` → `signature` (used verbatim) |
| `ground_truth_code` | `RECORDS.code` |
| `lineno` | `RECORDS.lineno` |
| `end_lineno` | `RECORDS.end_lineno` |

The script parses the official log's task rows (`task_id ran passed cumulative_pass`), checks every source ID against the log, and retains only rows with `ran=1` and `passed=1`. The three expected failures and final dependency counts are comparison assertions, not filtering rules. `results/ground_truth_validation.csv` contains all 230 tasks and their observed PASS result. The validation JSON records counts, missing fields, duplicates, failures, and dependency distribution after reading the saved dataset back.
