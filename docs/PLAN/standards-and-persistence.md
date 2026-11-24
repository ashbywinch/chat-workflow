# Standards, Verdicts and Persistence — implementation plan

The design is in [rules and authority](../reference/rules-and-authority.md) and
[document store](../reference/document-store.md). This page is the build order.

**Status:** the minimal path is next; the persistence and release machinery is
deferred until the evals show it is needed.

## Process decisions

- **Branch.** Work happens on a branch cut from `meta-workflow`. Nothing is built
  off `main` while [PR#17](https://github.com/ashbywinch/chat-workflow/pull/17)
  is open.
- **The existing feedback loop is the mechanism.** A failed validation is an
  error message fed back to the model, which continues the conversation. No new
  failure type, router, release, version store or prompt-hint mechanism is built.
  A meta-rule's text *is* its repair message.
- **A component is released when it is accepted at the end of its step.** There
  is no separate release action.

## Now

### 1. Rules about rules

- **Inputs:** the author's intent, exemplars and quality statements.
- **Outputs:** the meta-rules attached to the requirement-bearing model, so the
  existing validation feedback rejects an ambiguous requirement and the model
  continues the conversation.
- **Quality gate:** an ambiguous rule is rejected with a repair; a deterministic
  rule is written as a code check; a compound or example-bearing rule is
  rejected with the repair.

### 2. Prove it with the evals

- **Inputs:** the failures the suite produces, including a real truncated session
  from the original failure.
- **Outputs:** the previously failing evals green; a replay case that shows an
  ambiguous rule being rejected rather than guessed.
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
