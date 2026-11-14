# PLAN — core/components split

**Status:** Phase 1 landed on `core-split` (stacked on PR #17)
**Date:** 2026-11-14

The repo ships the framework, its CLI, its components, and their tests as one
package with one CI lane. This plan separates the core lane from the component
lanes so the core can be installed, tested, and evaluated without any component
package present, and so components can ship on their own cadence.

The plan is deliberately ordered: the mechanical package move comes first, then
the test kit, then the component packages, then the registry, then CI lanes.
Each phase ships on its own; no phase needs a later phase's output.

## Target layout

```
pyproject.toml                          # virtual workspace root: no [project]
uv.lock
packages/
  chat-workflow/
    pyproject.toml
    src/chat_workflow/
  chat-workflow-cli/
    pyproject.toml
    src/chat_workflow_cli/
  chat-workflow-components/             # module names unchanged until Phase 3
    pyproject.toml
    src/workflows/
  chat-workflow-testing/                # Phase 2
    src/chat_workflow_testing/
  chat-workflow-component-generator/    # Phase 3
    src/chat_workflow_component_generator/
  chat-workflow-acp/                    # ACP plan, Phase B
    src/chat_workflow_acp/
tests/                                  # one suite until Phase 2 de-tangles it
```

The root `pyproject.toml` carries the workspace members, ruff, pyrefly,
coverage, and the dev dependency group. Each package carries its own
`[project]` and build backend. Setuptools with `where = ["src"]` keeps the
build consistent with the pre-split repo.

## Dependency rules

```
chat-workflow (core)
  ├── chat-workflow-testing             (test-only dependency of every package)
  ├── chat-workflow-cli                 (composition root)
  ├── chat-workflow-components          (imports core)
  ├── chat-workflow-component-generator (imports core)
  └── chat-workflow-acp                 (imports core + registry; no concrete components)
```

- Core imports no other package.
- Testing may import core.
- Components and the generator may import core, never the CLI or ACP.
- CLI and ACP may import core and the registry, never a concrete component.
- Enforced by `tests/unit/test_architecture.py`, which scans production imports
  and maps each imported top-level module to its layer. It does not use
  archunitpython's layer API: that resolves absolute imports only from the
  project root, so the `src/` layout made every cross-package edge look
  external and the check passed vacuously.

## Phases

### Phase 1 — Move to `packages/` + `src/` (done)

- **Inputs:** the pre-split tree: three top-level modules and one root test
  suite.
- **Outputs:** the same three modules importable under unchanged names
  (`chat_workflow`, `chat_workflow_cli`, `workflows`) from a `src/` layout; the
  same CLI commands; the same tests, plus one guard test for the rewritten
  architecture check.
- **Operations:** virtual workspace root, one `pyproject.toml` per package,
  Makefile/lint/coverage/pyrefly/CI paths updated, stale egg-info removed,
  reinstall from the workspace root.
- **Quality gate:** `make test`, `make check`, and `make docs-check` green; no
  behavior change.

Phase 1 deviations, recorded so the later phases are not surprised:

- The test suite stays at root `tests/` as one suite. Distributing it needs the
  shared eval kit (`tests/conftest.py`, `tests/evals/helpers.py`,
  `tests/sample_code.py`) extracted first — that is Phase 2's job, and doing it
  here would stop the phase being a pure refactor.
- Only the three packages that hold code are created. The testing, generator,
  and ACP package skeletons arrive with the phases that give them code; an
  empty skeleton adds a pyproject that builds nothing.
- Path fixes that the move forced, all behavior-preserving: `docs-check` now
  runs `scripts/check_docs_links.py` (repo root from the script location, not
  from a dotted test module); `chat_workflow.config.default_config_path()`
  finds `config.json` by walking up instead of assuming a package depth;
  generated-code validation passes ruff the nearest `[tool.ruff]` config
  explicitly, because ruff stops at the nearest `pyproject.toml` and would
  otherwise silently drop the project's rule set; ruff's `known-first-party`
  is named explicitly so import grouping is stable across the `src/` layout.
- **Type checking has no baseline.** The committed `.pyrefly-baseline.json`
  and `scripts/pyrefly-lock.py` are gone; `make typecheck` requires zero
  diagnostics, so a package move can never again re-key a suppression file.
  The 73 baselined diagnostics were fixed instead: `@override` on real
  overrides, `cast(Any, f).attr` for attributes attached to function objects,
  `cast` where a Pydantic validator already guarantees the invariant, a float
  timeout type, `Self` on the Pydantic after-validator, and an `invalid_model()`
  test helper for the tests that construct models with required fields
  deliberately omitted.
- **Prompt-only bodies stay `...`.** An `@atomic_workflow_step` function is a
  declaration: its docstring is the conversation and its return annotation is
  the result model. pyrefly accepts a lone ellipsis as the body and rejects
  `pass`, a bare docstring, or `...` followed by `pass` — so the 14
  `# type: ignore[reportReturnType]` comments were unnecessary pyright
  leftovers and are gone, and `refine.py`'s trailing `pass` was the one real
  diagnostic. The author guide now teaches `...` instead of `pass`.
- The architecture test was rewritten to scan imports directly. archunitpython
  resolves absolute imports only from the project root (and its parent), so
  under `packages/<name>/src/` every cross-package edge resolved as external
  and the layer check passed vacuously — measured with an inverted rule: 39
  component→core violations before the move, 0 after. The replacement maps each
  imported top-level module to its layer and adds a guard test that fails if
  the expected cross-package edges disappear. archunitpython is no longer a
  dev dependency.
- Coverage denominator: 2495 statements before, 2520 after (the path fixes and
  the explicit scaffold bodies), 91% both — the measured module set is
  unchanged.

### Phase 2 — Extract `chat-workflow-testing` and de-tangle core tests

- **Inputs:** root `tests/`, component fixtures.
- **Outputs:** an installable test kit (`AgentIO`, `llm_judge`,
  `format_transcript`, `capture_on_failure`, `run_multi_turn_eval`,
  token/report helpers, timeout decorator, `FakeConfig`,
  `MockInstructorClient`) plus test components (`Criteria`, `Letter`).
- **Operations:** move the kit out of root `tests/`; move component fixtures to
  the components package; repoint core tests off `workflows.evaluation_criteria`;
  add the `client_factory` seam; add `resolve_config_path()`.
- **Quality gate:** `make test-core` and `make evals-core` pass in an isolated
  environment with no component package installed; the architecture test
  forbids core → components.

### Phase 3 — Split the component side

- **Inputs:** the component modules and the generator code in core.
- **Outputs:** `chat_workflow_components` (evaluation criteria, receptionist,
  directory, dispatch), `chat_workflow_component_generator` (the current
  `workflows/workflow` package plus `code_generator.py`).
- **Operations:** rename `workflows`; move `code_generator.py` out of core; add
  the receptionist and directory components; delete the broken `letter` stub;
  update entry points and imports.
- **Quality gate:** all package tests pass; dependency rules hold; `uv build`
  succeeds for every package.

### Phase 4 — Entry-point registry and CLI rewrite

- **Inputs:** component manifests and installed packages.
- **Outputs:** `chat_workflow.components` (manifest, catalog, entry protocol,
  local loader, change events); CLI discovery through the catalog; `component
  list`, `component run`, `component promote`.
- **Operations:** register components via
  `[project.entry-points."chat_workflow.components"]`; preserve command names
  through the `commands` mapping.
- **Quality gate:** the CLI lists and runs components without importing concrete
  component modules; a fake entry point and a local manifest are both tested.

### Phase 5 — CI lanes and change-gated full suite

- **Inputs:** the changed-path set of a PR.
- **Outputs:** per-lane unit + eval jobs with a cheap cross-side smoke; a
  change-gated full suite that skips when nothing eval-relevant changed.
- **Operations:** `make ci-plan` maps paths to lanes; `uv run --isolated
  --package …` enforces lane isolation.
- **Quality gate:** the plan is right for core-only, component-only, and shared
  diffs; the full suite skips on docs-only changes.

### Phase 6 — Release independence (deferred until publishing)

- **Inputs:** the built wheels.
- **Outputs:** independently versioned packages; components pin a compatible
  core range.
- **Operations:** `uv build` per package; install the components/generator
  wheels against the built core wheel in a clean venv.
- **Quality gate:** built wheels install cleanly and pass unit + one smoke eval.
