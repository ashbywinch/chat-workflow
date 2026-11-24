# Standards, Verdicts and Persistence — implementation plan

The design is in [rules and authority](../reference/rules-and-authority.md) and
[document store](../reference/document-store.md). This page is the build order.

**Status:** the minimal path is next; the persistence and release machinery is
deferred until the evals show it is needed.

## Process decisions

- **Branch.** Work happens on a branch cut from `meta-workflow`. Nothing is built
  off `main` while [PR#17](https://github.com/ashbywinch/chat-workflow/pull/17)
  is open.
- **One PR at a time.** The minimal path below is one PR: the meta-rules, the
  message improvements, and the eval that proves the loop.
- **The existing feedback loop is the mechanism.** A failed validation is an
  error message fed back to the model, which continues the conversation and asks
  the author when it needs to. No new failure type, router, release or version
  store is built now.
- **A component is released when it is accepted at the end of its step.** There
  is no separate release action.

## Now

### 1. Rules about rules in the requirement step

- **Inputs:** the author's intent, exemplars and quality statements.
- **Outputs:** the rule-quality meta-rules as the requirement set's validation
  rules, and one prompt hint: if a rule is ambiguous, ask the author which
  reading they mean before finishing; never choose one.
- **Quality gate:** an ambiguous rule produces a question and the step does not
  finish; a deterministic rule is written as a code check; a compound or
  example-bearing rule is rejected with the repair.

### 2. Messages that repair, in the existing loop

- **Inputs:** the failures the suite already produces.
- **Outputs:** every rejection names the offending value and a concrete repair —
  the stage that repeats its step, a placeholder, a type expression in prose, a
  success that asks a question. These are the messages the model is given when it
  is asked to continue.
- **Quality gate:** the failing eval passes after the message change, without a
  new mechanism.

### 3. Prove it with the evals

- **Inputs:** the failures the suite produces, including a real truncated
  session from the original failure.
- **Outputs:** the previously failing evals green; a replay case that shows an
  ambiguous rule being questioned rather than guessed.
- **Quality gate:** the conversation suite passes; no rule change is made
  silently.

## Deferred

Recorded in the design notes, not built now. Revisit only if the feedback loop
proves insufficient:

- requirement ids and versions, and the document store (`document-store`,
  `document-store-files`);
- persisting requirements, artefacts, verdicts and code;
- the rework surface beyond the existing conversation;
- releases and amendments as separate machinery.

## Out of scope

The enterprise adapter, multi-user operation, permissions, search/RAG and
retention policy.
