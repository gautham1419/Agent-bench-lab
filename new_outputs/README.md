# Raw Experiment Logs

This directory contains the raw output of all 216 agent-environment runs.

## Data Correction Note

One folder has a stale leading timestamp that does not reflect the actual
experiment start time:

```
2026-03-21-23-16-36_-ollama-ministral3-8b-reasoning-q8_0-run2/
```

This folder was created on **2026-03-21** but the run did not execute until
**2026-04-07** (verified from `runs.jsonl` task-completion timestamps). Using
the folder name as the start time naïvely produced a wall-clock of
**1,444,141 s (≈ 17 days)**, which is a data-pipeline artefact, not a
runtime anomaly. The run itself completed normally.

The fix is implemented in `statistical_tests/deployment_metrics.py` via a
two-pass strategy:

1. For every *valid* run (folder timestamp within 1 hour of first-task
   completion) compute **task-1 duration** = first-task-timestamp −
   folder-timestamp.
2. Build a per-`(model, size, quant)` average task-1 duration from all valid
   runs of the same configuration.
3. For the affected run, estimate the true start as:
   `corrected_start = first_task_timestamp − avg_task1_duration`

The corrected wall-clock for this run is **≈ 5,472 s / mean-task ≈ 40 s**,
consistent with replicate 3 (5,314 s / 39 s). All 24 model-domain pairs are
therefore used in every paired statistical test in the paper.

## Folder Naming Convention

```
YYYY-MM-DD-HH-MM-SS_[model]-[size]-[quant]-run[N]/
```

Examples:
- `2026-03-10-11-21-44_ollama-qwen3-4b-t-q4_k_m-run1/` — Qwen3-4B at Q4_K_M, first replicate
- `2026-04-01-17-04-02_ollama-qwen3-8b-bf16-run2/` — Qwen3-8B at BF16, second replicate

The leading timestamp encodes the experiment start time and is used to compute run-level wall-clock duration.

## Folder Contents

```
TIMESTAMP_model-run/
├── config.yaml            # Exact agent and task configuration for this run
├── resource_metrics.json  # Hardware telemetry (GPU/CPU energy, memory)
└── agent-name/
    └── domain-std/
        ├── runs.jsonl     # Per-task outcomes (one JSON line per task)
        └── error.jsonl    # System-level errors (tasks that could not start)
```

## `runs.jsonl` — Per-task outcomes

Each line is a complete record for one task attempt:

```json
{
  "index":  42,
  "error":  null,
  "info":   null,
  "output": {
    "index":  42,
    "status": "completed",
    "result": {
      "finish":   true,
      "reward":   1,
      "status":   "completed",
      "messages": [ ... ],
      "metrics":  { "score": 1 }
    },
    "history": [ ... ]
  },
  "time": { "timestamp": 1773121938273, "str": "2026-03-10 11:22:18" }
}
```

| Field | Type | Meaning |
|---|---|---|
| `index` | int | Task ID (0-based integer, fixed across all runs of the same domain) |
| `output.result.reward` | 0 or 1 | Task outcome (1 = success) |
| `output.result.metrics.score` | 0 or 1 | Same as reward (alternative field used by some domains) |
| `output.status` | string | Terminal status — see below |
| `output.result.messages` | list | Full agent-environment conversation |
| `time.timestamp` | int | Unix milliseconds — task completion time |

### Task status values

| Status | Meaning | Counted as |
|---|---|---|
| `completed` | Task reached a terminal state | Success if reward > 0, else completed_failure |
| `task limit reached` | Agent exceeded the 200-turn limit | TLE failure |
| `task error` | Task environment error | task_error failure |
| `agent invalid action` | Agent emitted an unparseable action | IA failure |

Invalid-format (IF) failures are detected post-hoc: a completed task with empty or malformed `tool_calls` arguments.

## `error.jsonl` — System-level errors

Tasks where the session could not be established (e.g., Docker container crash, network failure). These count toward the denominator of success rate.

```json
{ "index": 16, "error": "INTERACT_FAILED", "info": "{\"message\":\"session not found\"}" }
```

| Error type | Meaning |
|---|---|
| `INTERACT_FAILED` | Agent-environment interaction failed mid-session |
| `START_FAILED` | Task environment failed to start |
| `AGENT_FAILED` | Agent process crashed |

## `resource_metrics.json` — Hardware telemetry

Recorded by a monitoring process that polls hardware sensors at 1-second intervals:

```json
{
  "cpu_avg":         3.02,
  "cpu_peak":        14.3,
  "ram_avg":         589.0,
  "ram_peak":        589.9,
  "gpu_util_avg":    5.12,
  "gpu_util_peak":   44,
  "gpu_mem_avg":     4912.9,
  "gpu_mem_peak":    4973.4,
  "gpu_power_avg":   68.5,
  "gpu_energy_joules": 109242.97,
  "cpu_energy_joules": 48648.75,
  "total_energy_joules": 157891.72
}
```

| Field | Unit | Measured by |
|---|---|---|
| `gpu_mem_peak` | MiB | pynvml |
| `gpu_power_avg` | W | pynvml |
| `gpu_energy_joules` | J | numerical integration of gpu_power_avg × 1 s |
| `cpu_energy_joules` | J | pyRAPL (RAPL interface) |
| `total_energy_joules` | J | gpu_energy + cpu_energy |

**Note on tokens/s:** The Ollama API response is captured via a `return_format` template that extracts only the decoded text. Prompt and completion token counts are not recorded, so tokens/s throughput cannot be computed from these logs.

## Task Index Stability

The `index` field is stable across all runs of the same domain. AgentBench draws tasks from the same data files in a fixed order, so `index=42` in a Q4_K_M run refers to exactly the same problem as `index=42` in the corresponding BF16 run. This property is exploited by the TOST equivalence analysis to pair task outcomes across precision levels.

## Coverage

| Model | Size | Quants | Domains | Replicates | Runs |
|---|---|---|---|---|---|
| Qwen3 | 4B | Q4_K_M, Q8_0, F16 | OS, DB, WS, ALF | 3 | 36 |
| Qwen3 | 8B | Q4_K_M, Q8_0, BF16 | OS, DB, WS, ALF | 3 | 36 |
| Ministral3 | 3B | Q4_K_M, Q8_0, BF16 | OS, DB, WS, ALF | 3 | 36 |
| Ministral3 | 8B | Q4_K_M, Q8_0, BF16 | OS, DB, WS, ALF | 3 | 36 |
| DeepSeek-R1-Qwen | 1.5B | Q4_K_M, Q8_0, F16 | OS, DB, WS, ALF | 3 | 36 |
| DeepSeek-R1-Qwen | 7B | Q4_K_M, Q8_0, F16 | OS, DB, WS, ALF | 3 | 36 |
| **Total** | | | | | **216** |
