# PLAN — chat-workflow

Each phase delivers a usable capability on its own; no phase depends on a
later one.

## Phase 0 — Framework skeleton (done)

- **Inputs**: a domain model and a prompt.
- **Outputs**: `@atomic_workflow`, turn loop, `AgentResponse`, typed result
  or named exception; `@composite_workflow`; CLI discovery.
- **Operations**: `make` creates the venv and installs the package.
- **Quality gate**: unit tests with a fake instructor client; `make lint`.

## Phase 1 — Validation and artifacts (done)

- **Inputs**: `Validation(...)` annotations, `Blob(...)` fields, business
  rules on the Pydantic model.
- **Outputs**: LLM-validated values, materialised files, `LLMValidated` and
  `BlobSyncMixin`.
- **Operations**: nothing new — validation is part of the atomic flow.
- **Quality gate**: unit tests for annotation parsing, rule aggregation and
  blob sync.

## Phase 2 — Real-model evals (done)

- **Inputs**: a persona prompt, judge rules, a live model.
- **Outputs**: multi-turn eval tests, transcripts on failure, token/cost
  report.
- **Operations**: `make evals`, `make evals-smoke`, `make evals-incremental`
  under a timeout wrapper.
- **Quality gate**: evals green through Cloudflare AI Gateway; CI uploads
  transcripts on failure.

## Phase 3 — Generated workflows (in progress)

- **Inputs**: a process description and example components.
- **Outputs**: generated component source code, static verification,
  self-validation evals.
- **Operations**: the workflow-generation CLI and its evals.
- **Quality gate**: generated code passes ruff and the component contracts.
  This phase is why the current branch is `meta-workflow`; it is not a
  prerequisite for phases 0–2.

## Phase 4 — Scale and packaging (future)

- **Inputs**: caller demand.
- **Outputs**: packaging and concurrency decisions, not yet specified.
- **Operations**: a release process.
- **Quality gate**: coverage floor raised, versioned releases.

## Quality gates in force

| Gate | Command | Blocks |
|---|---|---|
| Lint | `make lint` | a push (pre-commit runs it) |
| Types | `make typecheck` | a push (pre-push runs `make check`) |
| Unit tests | `make test` | a merge |
| Docs links | `make docs-check` | a merge |
| Architecture | `tests/unit/test_architecture.py` | a merge |
| Coverage | `make coverage` | a merge below the CI floor |
| Evals | `make evals` | a merge (CI runs the smoke subset on PRs; the full suite runs weekly/manual) |
