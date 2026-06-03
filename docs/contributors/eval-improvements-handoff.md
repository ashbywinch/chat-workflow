# Eval Improvements — Handoff Document

## Before starting

Read the contributor docs in `docs/contributors/`, especially:
- `testing.md` — testing strategy, eval design principles, cost optimization
- `writing-documentation.md` — context efficiency, one-topic-per-file

Also read `AGENTS.md` for the decision tree and `README.md` for project overview.

## Goals

1. **All single-workflow evals pass consistently** (no flaky tests)
2. **All composite evals pass consistently** (same)
3. **Reasonable time/cost** — currently ~$0.08-0.19 per full single suite, ~$0.10 per full suite
4. **Cost tracking infrastructure works** — `CHAT_WORKFLOW_EVAL_REPORT=1`, `scripts/eval_report.py`
5. **Incremental evals work correctly** — `make evals-incremental` uses transitive dependency analysis

## Where we are

**Single evals (`tests/evals/single/`):** 41 tests, ~12 flaky failures.

The failures fall into two categories:

### Category A: Agent asks instead of proposing
The core issue: workflow prompts tell the agent to "propose" but the model defaults to asking Socratic questions. This affects:

- `test_domain_exploration` — agent repeats field proposals after user confirms
- `test_design_from_domain_spec` — agent asks about each field individually
- `test_gather_interaction_context` — agent asks multi-part questions instead of single probes
- `test_output_proposes_structure` — agent doesn't propose meeting structure from knowledge
- `test_output_warm_open` — agent jumps into questions without greeting
- `test_resource_no_forced_field_mapping` — agent doesn't propose from knowledge

**Key insight from user:** The prompts should use information from *previous conversation turns* (passed as parameters like `analysis`, `domain_spec`, `responsibilities`), not just "common knowledge." The agent already has context — it should propose based on that context.

### Category B: Repetition loops
The agent asks the same question after the user answers. This affects:

- `test_output_adapts_to_suggestions` — repeats question after pushback
- `test_anything_else_pattern` — asks about stages again after user described them
- `test_no_premature_structuring` — returns None (workflow timeout)
- `test_synthesizes_honestly` — repeats "tell me more" after user answers

The root cause: the `_gather_notes` and `_generate_from_notes` prompts in `process_definition.py` are too minimal. The 58-word original was cut even shorter and lost the behavioral guardrails.

### Category C: Judge rule quality
The `CONFUSION_JUDGE_RULES` and `JARGON_FREE_RULES` were previously word-blacklists that caused false positives. They've been rewritten to use a "match the user's language" philosophy but the LLM judge still sometimes misinterprets.

## What's been done

### Prompt conciseness
| Prompt | Before | After | Savings |
|--------|--------|-------|---------|
| ComponentStructure.design | 640 words | ~170 words | ~73% |
| ComponentDomainSpec.explore | 460 words | ~130 words | ~72% |
| ComponentInteractionContext.gather | 569 words | ~170 words | ~70% |
| Deliverable.generate_from_chat | 119 words | ~90 words | ~24% |
| Resource.generate_from_chat | 209 words | ~80 words | ~62% |
| _gather_notes | 58 words | ~70 words (rewritten) | — |
| _generate_from_notes | 140 words | ~70 words | ~50% |

### Infrastructure
- `config.json` supports presets (`active` + `presets`) for easy provider switching
- `model_supports_tools` flag controls `Mode.TOOLS` vs `Mode.JSON`
- Custom `api_base` + `api_key_env` for OpenAI-compatible endpoints (OpenCode Go)
- `@timeout` decorator + global litellm callback tracks ALL eval test costs
- `scripts/eval_report.py` generates grouped reports with per-test time/token/cost
- `CHAT_WORKFLOW_SAVE_TRANSCRIPT=1` saves transcripts on success too
- Evals moved to `single/` and `composite/` subdirectories for clarity
- `affected_evals.py` uses transitive BFS dependency tracking
- igraph installed — community detection no longer falls back

### Judge rules
- `CONFUSION_JUDGE_RULES` now uses "stay in user's world" philosophy, not word blacklist
- `JARGON_FREE_RULES` same approach — match user language, not field names

## What remains

### 1. Make all single evals pass consistently (~12 flaky failures)

The core fix: workflow prompts should use conversation context (previous parameters) to propose, not ask. For example:

- `Resource.generate_from_chat` receives `analysis` (ProcessDefinition) and `outputs` — use those to propose what the user starts with
- `Deliverable.generate_from_chat` receives `analysis` — use it to propose relevant outputs
- The prompts need to say "Based on what you already know about their process, propose..."

### 2. Make all composite evals pass

The `composite/` evals (workflow_evals, end_to_end, codegen, pipeline, debug_streaming) haven't been run recently. They may have pre-existing failures or new ones from prompt changes.

### 3. Run full baseline and compare costs

After fixing failures, run the full single suite and compare costs using `scripts/eval_report.py`. Target: no increase in total cost vs baseline.

### 4. Verify incremental evals

The `make evals-incremental` target should detect which evals are affected by changes. After the recent moves to `single/` and `composite/`, verify it still works:
```bash
python scripts/affected_evals.py --git-base origin/main --verbose
```

### 5. CI workflow

The CI runs `make evals` (full suite, not incremental). The `.github/workflows/tests.yml` was updated with `OPENCODE_GO_EVALS_API_KEY` secret. Verify CI passes after all fixes.

## Key files

| File | Purpose |
|------|---------|
| `tests/evals/single/test_*.py` | Single-workflow evals (fix these first) |
| `tests/evals/composite/test_*.py` | Multi-workflow evals |
| `tests/evals/helpers.py` | `run_multi_turn_eval`, `EvalStats`, `_token_counter_callback` |
| `tests/conftest.py` | `@timeout` decorator with token tracking |
| `chat_workflow/llm_interaction.py` | `get_client()` with api_base, api_key_env, model_supports_tools |
| `chat_workflow/config.py` | Preset-based config resolution |
| `scripts/eval_report.py` | Grouped cost/time report generator |
| `scripts/affected_evals.py` | Transitive incremental eval detection |
| `scripts/run_with_timeout.py` | Hard timeout wrapper for eval suite |
| `Makefile` | `evals-incremental` target (1800s timeout) |
| `config.json` | Provider presets (openrouter, opencode-go, deepseek-or) |

## Running evals

```bash
# Quick: single workflow tests only
make test-unit && .venv/bin/python -m unittest discover tests/evals/single/

# With cost tracking
CHAT_WORKFLOW_EVAL_REPORT=1 .venv/bin/python -m unittest discover tests/evals/single/

# View cost report
python scripts/eval_report.py && cat test-results/cost-report.txt

# Full incremental suite
make evals-incremental

# Check affected evals
python scripts/affected_evals.py --git-base origin/main --verbose
```

## Current cost estimate
- Single suite: ~$0.08-0.10 (36-41 tests, ~7-11 min)
- Full suite: ~$0.19 (49 tests tracked, ~13 min)
- Most expensive single test: `test_full_workflow_pipeline` at ~$0.07
