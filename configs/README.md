# Experiment Configuration Files

Configuration files used to run the experiments. Each file is referenced by the `config.yaml` inside the corresponding `new_outputs/` folder.

## Directory Structure

```
configs/
├── eval_config.yaml         # Evaluation pipeline settings
├── start_task.yaml          # Task server startup configuration
│
├── agents/                  # Agent (LLM) definitions
│   ├── ollama-common-chat.yaml  # Ollama chat format (Ministral3, DeepSeek)
│   ├── ollama-qwen-chat.yaml    # Ollama chat format with Qwen-specific settings
│   ├── openai-chat.yaml         # OpenAI-compatible API agents
│   └── ...
│
├── tasks/                   # Task environment definitions
│   ├── os.yaml              # OS Interaction task config
│   ├── dbbench.yaml         # Database task config
│   ├── webshop.yaml         # WebShop task config
│   ├── alfworld.yaml        # ALFWorld task config
│   └── task_assembly.yaml   # Imports all task definitions
│
└── assignments/             # Per-domain run scripts
    ├── definition.yaml      # Imports all agent and task definitions
    ├── os_only.yaml         # Run one model on OS domain
    ├── dbbench_only.yaml    # Run one model on DB domain
    ├── alfworld_only.yaml   # Run one model on ALFWorld domain
    └── webshop_only.yaml    # Run one model on WebShop domain
```

## Running an Experiment

1. Edit the relevant `assignments/` file to uncomment the desired model
2. Set the `output` path to include the run timestamp and label
3. Start the Docker environment for the target domain
4. Run the assigner:

```bash
python -m src.assigner --config configs/assignments/os_only.yaml
```

## Model Name Mapping

| Config name | Ollama model tag | Quant |
|---|---|---|
| `ollama-qwen3-4b-t-q4_k_m` | `hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M` | Q4_K_M |
| `ollama-qwen3-4b-t-q8_0` | `hf.co/unsloth/Qwen3-4B-GGUF:Q8_0` | Q8_0 |
| `ollama-qwen3-4b-t-f16` | `hf.co/unsloth/Qwen3-4B-GGUF:F16` | F16 (=BF16) |
| `ollama-qwen3-8b-q4_k_m` | `hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M` | Q4_K_M |
| `ollama-ministral3-3b-reasoning-q4_k_m` | `hf.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF:Q4_K_M` | Q4_K_M |
| `ollama-ministral3-8b-reasoning-bf16` | `hf.co/unsloth/Ministral-3-8B-Instruct-2503-GGUF:BF16` | BF16 |
| `deepseek-r1-qwen-1.5b-q4_k_m` | `hf.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M` | Q4_K_M |
| `deepseek-r1-qwen-7b-f16` | `hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:F16` | F16 (=BF16) |

See the `config.yaml` inside any `new_outputs/` folder for the exact model tag used in that run.

## Inference Settings (fixed across all runs)

| Parameter | Value |
|---|---|
| Temperature | 0 (deterministic) |
| Max turns | 200 |
| Concurrency | 32 tasks in parallel |
| Output format | JSON tool calls |
| Retry on format error | 1 retry |
