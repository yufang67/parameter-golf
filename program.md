# Program

## Goal

Optimize for the OpenAI Model Craft Challenge: Parameter Golf.

**Primary metric:** `val_bpb`

**Direction:** minimize

**Definition:** tokenizer-agnostic bits-per-byte on the fixed FineWeb validation split

**Primary source of truth:** this file and the evaluation path inside `train_gpt*.py`

For headline record claims, use the following statistical improvement rule:

- A new SOTA should beat the current record by at least `0.005` nats at `p < 0.01`.
- This requirement is for leaderboard submissions, not for ordinary local iteration.

## Budgets

Two budgets matter simultaneously:

1. **Artifact budget**
    - Total counted bytes must satisfy: the run script plus the compressed model must be
       `len(the Python entrypoint used for the run) + len(compressed_model) <= 16_000_000`
   - This is decimal megabytes, not MiB.

2. **Current working setup**
   - This repo is currently operated primarily through the local `.venv` environment.
   - The active multi-GPU workflow in this repo is primarily `4x A100`, as reflected in `command.txt`.
   - Multi-GPU experiments in this repo should target the current `4x A100` workflow unless explicitly changed.

## Constraints

### What the agent should optimize

- Lower `val_bpb` without violating artifact-size or runtime requirements.
- Prefer changes that survive round-trip serialization and post-training quantization.
- Treat training speed, evaluation speed, and compressed size as first-class metrics, not afterthoughts.

### What the agent can edit
- create new `train_gpt_*.py` files based on existing SOTA solutions under the `records` folder
- modify existing `train_gpt_*.py` files
- Repo-local experiment notes such as `strategy.md`, `journal.md`, and `results.tsv` may be updated.
- Helper scripts or docs may be edited when needed to support the current workflow, but avoid unrelated churn.
- install third-party libraries to accelerate training, compression, or quantization

### What the agent must not do

- Do not change the working rules defined by this file without explicit direction.
- Do not rely on network access, external downloads, or training-data access during evaluation.
- Do not claim a leaderboard-worthy result without evidence for runtime, artifact size, and reproducibility.
- Do not introduce large complexity for marginal gains unless the gain is clearly worth it.

### Simplicity criterion

All else equal, prefer the simpler mechanism. In this challenge, complexity has at least four costs:

- More code bytes counted toward the 16MB cap
- More places for serialization or evaluation bugs
- More difficulty reproducing a result
- More friction when packaging a records submission

Keep a complex change only if it pays for itself in `val_bpb`, speed, size, or robustness.

## Files

| File | Who edits | Purpose |
|------|-----------|---------|
| `program.md` | Agent/Human | This operating spec for this repo |
| `train_gpt.py` | **Agent** | Original CUDA training, eval, quantization, serialization |
| `train_gpt_04_09.py` | **Agent** | Training, eval, quantization, and serialization based on the 2026-04-09 solution, with architecture changes |
| `test_run_1311.py` | **Agent** | Local script related to experiments around PR 1311 |
| `train_gpt_2026-04-05.py` | **Agent** | Training, eval, quantization, and serialization based on the 2026-04-05 solution with full residual attention |
| `train_gpt_new.py` | **Agent** | Training, eval, quantization, and serialization for repo-local experiments derived from the original `train_gpt.py` |
| `strategy.md` | **Agent** | Current search phase, hypotheses, priorities |
| `journal.md` | **Agent** | Concise experiment narrative and lessons learned |
| `results.tsv` | **Agent** | Structured run log |
| `records/...` | **Agent** | Submission packaging area for accepted runs |

Read the relevant files before making changes. This file defines the current repo workflow.

## Setup

Before starting a serious experiment cycle:

1. Read this file, `strategy.md`, `journal.md`, and `results.tsv`.
2. Two dataset roots are available: `sp1024` under `./data` and `sp8192` under `./data2`.
3. Decide whether the run is:
   - a local smoke test,
   - a single-GPU CUDA experiment,
   - a 4xA100 CUDA run.
4. Choose a clear `RUN_ID` and log the intent before changing code.

If prerequisites are missing, stop and report the exact command needed to fetch them.

Example command lines are collected in `command.txt`.

## Experiment loop

Run a disciplined loop:

1. **Read state**
   - Check `strategy.md`, `journal.md`, and `results.tsv`.
   - Check the strongest relevant results under the `records` folder, because they are validated reference points and practical baselines.
   - Identify the potential improvements.
   - Identify the best-known tradeoff among `val_bpb`, artifact size, and runtime.

2. **Write one hypothesis**
   - State one concrete change and why it might help.
   - Prefer one dominant variable or implementation per run when possible.

3. **Implement**
   - Create a new file or edit the smallest set of existing files necessary.
   - Keep submission constraints in mind while coding, especially counted code size.

4. **Run**
   - The environment is already set up at `.venv`.
   - Execute the appropriate smoke test or training run.
   - Capture logs without flooding the working context.

5. **Read results**
   - Record at minimum: `val_loss`, `val_bpb`, compressed size, total runtime, and any notable eval details.

6. **Decide**
   - Keep, discard, or mark as inconclusive.
   - A `val_bpb` win that breaks size or runtime is not a full win.

7. **Log**
   - Append structured results to `results.tsv`.
   - Add a short narrative entry to `journal.md`.
   - Update `strategy.md` if the search direction changed.

## Keep / discard rules

- **Keep** if `val_bpb` improves and the change still respects the relevant size/runtime constraints.
- **Keep** if `val_bpb` is flat but the code is smaller, faster, or more reliable.
- **Discard** if `val_bpb` worsens without compensating gains elsewhere.
- **Discard** if a gain depends on violating the challenge rules.
- **Crash** if the run fails, the serialized round trip fails, or the reported metrics are invalid.

Treat a result as incomplete if any of these are missing for a candidate submission:

- compressed model size
- counted artifact size
- wallclock runtime
- final round-trip evaluation metric

## Phases

Track the current phase in `strategy.md`.

1. **Baseline**
   - Reproduce the current script as-is.
   - Establish reference `val_bpb`, runtime, and artifact size.

2. **Architecture search**
   - Explore model structure, recurrence, tokenizer choice, optimizer grouping, evaluation tricks, and serialization-aware changes.
   - Some examples:
     - linear attention, gated attention
     - Gemma-style architecture ideas
     - heterogeneous stacks instead of uniform layers: cheaper early layers and stronger late layers, such as MLP-heavy early blocks, attention/XSA-heavy late blocks, depth-varying MLP width, depth-varying KV heads, or depth-varying positional treatment
     - sliding-eval-aware training: add training perturbations that better match the real evaluation protocol, such as random context truncation, random scoring stride exposure, or window-boundary perturbations, to improve sliding-window BPB directly rather than only plain teacher-forced loss
     - tokenizer and embedding-factorization co-design: sweep tokenizer size jointly with factored tied embeddings, MLP width, and quantization regime instead of treating tokenizer choice as independent from model budget allocation
     - alternating local sliding-window and global full-context attention layers
     - dual RoPE configurations: standard RoPE for sliding layers and proportional RoPE for global layers to enable longer context
     - per-layer embeddings: a second embedding table that feeds a small residual signal into every decoder layer

3. **Systems and compression tuning**
   - Improve throughput, quantization, packing, and code-size efficiency.

4. **Hyperparameter refinement**
   - Narrow onto promising structures and tune learning rates, schedule, weight decay, validation cadence, and related knobs.

5. **Ablation and packaging**
   - Remove unnecessary complexity.
   - Prepare a self-contained `records/...` submission candidate with reproducible logs.

## Logging format

Use `results.tsv` as a tab-separated run ledger. Include enough fields to compare quality, size, and speed.

Recommended header:

```text
date	run_id	commit	track	data_variant	tokenizer	val_loss	val_bpb	artifact_bytes	compressed_model_bytes	wallclock_s	status	description
```

Field intent:

1. `date` — ISO date
2. `run_id` — the run identifier used at launch
3. `commit` — short git hash when applicable
4. `track` — `local`, `single_gpu`, `4xA100`, or `non_record`
5. `data_variant` — for example `sp1024` or `sp8192`
6. `tokenizer` — tokenizer file or shorthand
7. `val_loss` — final validation loss if available
8. `val_bpb` — primary score
9. `artifact_bytes` — counted submission bytes
10. `compressed_model_bytes` — compressed model only
11. `wallclock_s` — elapsed seconds for the run
12. `status` — `keep`, `discard`, `crash`, or `inconclusive`
13. `description` — short explanation of the change

## Journaling

After each meaningful run, append a short entry to `journal.md` covering:

- what changed
- why it was worth testing
- the observed `val_bpb`, size, and speed outcome
- what to try next

Keep entries concise and decision-oriented.

## Submission checklist

Before calling a run submission-ready, verify all of the following:

1. The model reproduces from the saved code and compressed artifact.
2. The total counted artifact is `<= 16_000_000` bytes.
3. The claimed runtime matches the intended challenge track.
4. `val_bpb` is computed correctly for the chosen tokenizer/data path.
5. The records folder contains the required metadata, logs, and runnable code for the current workflow.

## Safety

- If a run crashes, log it explicitly.
- If `val_bpb` or size accounting looks suspicious, assume the result is invalid until verified.
- Never delete prior experiment logs just because a result is worse.
- Do not represent exploratory wins as strong evidence without the required validation.
