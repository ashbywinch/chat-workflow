# Standards, Verdicts and Persistence — implementation plan

The design is in [rules and authority](../reference/rules-and-authority.md) and
[document store](../reference/document-store.md). This page is the build order.

**Status:** next, not started. Scope: one user, present, authoring a component.

## Process decisions

- **Branch.** Work happens on a branch cut from `meta-workflow`, not from `main`.
  [PR#23](https://github.com/ashbywinch/chat-workflow/pull/23) (the core split) is
  stacked on the same branch and will move the new packages under `packages/`;
  that move is file relocation, not redesign.
- **Package layout.** New packages live at the repository root until #23 lands:
  `document-store` (contract) and `document-store-files` (simple
  implementation). `document-store-enterprise` is not built in this phase.
  One class per module; packages organise them.
- **One PR.** The five items below are interdependent — a store with nothing to
  store, a router with no verdicts — so they land as one PR with one commit per
  item. If review prefers smaller changes, the split point is after item 2.
- **No persistence in the library.** The library gains only rule identity and the
  typed failure. It imports no store package and writes no file.
- **No new capture.** The eval of the eval replays a failure record the harness
  already writes.
- **Build for the split, not around it.** Nothing in the library imports the
  component maker; the component maker imports the store contract and the simple
  implementation.

## Work items

### 1. Rule identity and the typed failure (library)

- **Inputs:** `_validation_rules` strings; the judge's verdicts.
- **Outputs:** `Rule(id, version, text)`; an explicit semantic evaluation that
  runs after the object is constructed; `RequirementFailed` carrying the verdicts
  and the attempted artifact. Structural validation and its retry path are
  unchanged.
- **Operations:** none for the user.
- **Quality gate:** a semantic failure raises with rule ids and versions; a
  structural failure still reaches the model as a retry; the existing suite
  unchanged.

### 2. Storage contract and the simple implementation

- **Inputs:** the record shape and four operations in the store note.
- **Outputs:** `document-store` (`Document`, `DocumentType`, `DocumentStore`,
  `Release`, errors); `document-store-files` (`LocalDocumentStore` over
  `.workflow-store/`).
- **Operations:** `put`, `get`, `find`, `release`; configurable path.
- **Quality gate:** round-trip and versioning; a release is an immutable set of
  `uuid@version`; refs compose to arbitrary depth; the architecture test shows no
  library→store import.

### 3. Requirements as documents

- **Inputs:** the component's domain spec and checks.
- **Outputs:** the requirement record — intent, statements, one good and one bad
  exemplar, scope, owner, version — written to the store; checks carry
  `requirement_id`; generated code carries the same ids.
- **Operations:** the domain-spec step asks for intent and the two exemplars.
- **Quality gate:** every check resolves to a requirement version; a component
  whose rule has no requirement is rejected at authoring time.

### 4. Router and rework step

- **Inputs:** a `RequirementFailed` and the facts the user gave.
- **Outputs:** `route()` returning fix-check / fix-prompt / regenerate / ask, with
  the two-attempt budget; the rework conversation step that applies a spec fix
  with a citation or asks the author one question; an amendment is a new version.
- **Operations:** the rework step runs in the authoring conversation.
- **Quality gate:** a check is only repaired when the contradiction is quoted; a
  requirement change always asks; the same artifact is re-evaluated under the new
  version.

### 5. The eval of the eval

- **Inputs:** an existing failure record (`events`, transcript, verdicts),
  truncated at the failure.
- **Outputs:** a deterministic replay test that asserts the route, plus one live
  end-to-end eval of the loop.
- **Operations:** the live eval runs in the conversation suite.
- **Quality gate:** the replay reaches the same conclusion as the original
  failure; the live eval passes.

## Out of scope

The enterprise adapter, multi-user operation, permissions, search/RAG and
retention policy. The store contract is shaped so they are additions, not
rewrites.

## Open questions

- How a check's contradiction of its requirement is proven to the evidence bar.
- How an amendment is recorded against the component so the next run uses it.
- Explicit saves versus a thin persistence wrapper in the component maker.
- Retention; the blob adapter; where the runnable code file lives.
