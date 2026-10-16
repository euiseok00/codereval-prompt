# Main Java prompt experiment

This directory contains the current CoderEval Java prompt experiment.

## Data flow

```text
CoderEval/CoderEval4Java.json
        +
CoderEval/CoderEval-Input4Models.zip (CEJavaRaw.jsonl)
        +
results/validation/GroundTruthJava.log
        |
        v
experiment/data/CEJavaPrompt.jsonl
        |
        v
experiment/data/CEJavaPromptJobs.jsonl
        |
        v
results/main/GenerateResultJava/Run1.jsonl
results/main/GenerateResultJava/Run2.jsonl
results/main/GenerateResultJava/Run3.jsonl
        |
        v
results/main/EvaluateResultJava/Run1.jsonl
results/main/EvaluateResultJava/Run2.jsonl
results/main/EvaluateResultJava/Run3.jsonl
        |
        v
results/main/AnalysisJava/
```

`CEJavaPrompt.jsonl` contains the 227 Java tasks whose ground-truth implementation passed the preserved CoderEval validation run. It keeps only the fields used by the experiment or analysis: `task_id`, `method_name`, `human_label`, `signature`, and `dependency_level`.

`CEJavaPromptJobs.jsonl` expands those 227 tasks into the eight Persona × Signature × CoT conditions, producing 1,816 jobs. The same 1,816 jobs are executed independently three times.

## Conditions

| Condition | Persona | Signature | CoT |
| --- | --- | --- | --- |
| C1 | OFF | OFF | OFF |
| C2 | OFF | ON | OFF |
| C3 | OFF | OFF | ON |
| C4 | OFF | ON | ON |
| C5 | ON | OFF | OFF |
| C6 | ON | ON | OFF |
| C7 | ON | OFF | ON |
| C8 | ON | ON | ON |

## Model settings

- Model: `gpt-5.6-sol`
- Reasoning effort: `none`
- Temperature: `0.6`
- Independent runs: `3`
- Primary correctness metrics: `Pass@1`, `Pass@3`

## Commands

```bash
python experiment/scripts/BuildCEJavaPrompt.py
python experiment/scripts/RunCEJavaPrompt.py plan

python experiment/scripts/RunCEJavaPrompt.py generate --run 1 --confirm-full-run
python experiment/scripts/RunCEJavaPrompt.py evaluate --run 1
python experiment/scripts/RunCEJavaPrompt.py generate --run 2 --confirm-full-run
python experiment/scripts/RunCEJavaPrompt.py evaluate --run 2
python experiment/scripts/RunCEJavaPrompt.py generate --run 3 --confirm-full-run
python experiment/scripts/RunCEJavaPrompt.py evaluate --run 3

python experiment/scripts/AnalyzeCEJavaPrompt.py
```

The prior one-run experiment and its analysis are preserved under `experiment/archive/previous_experiment/` and `results/archive/previous_experiment/`.
