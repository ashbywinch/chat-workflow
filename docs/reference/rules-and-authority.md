# Rules and Authority

A rule is a statement of what "good" means for an artifact. The kinds of rule
and the principles behind them are in the PRD's
[Rules about rules](../PRD.md#rules-about-rules). This note explains how each
kind is evaluated, what a failure means, and who may change the standard.

**Status:** proposal. Scope: one user, present, actively authoring a component.
The absent-author case is out of scope here.

## The problem

The conversation suite surfaced failures that were not framework defects:

- a meeting with no decisions failed "every decision has an owner";
- a description failed "specific enough";
- a generated rule said "Location must be Zoom", because one meeting was on Zoom.

Each is a disagreement about what "good" means, not about the conversation, the
protocol, or the model. The current architecture answers every one of them the
same way: retry the model. A retry asks for a *different artifact*. It cannot
resolve a rule whose meaning is unsettled, so it either spends the budget or
fabricates a value. Either way the standard has been changed silently, by a
model, with no author.

## Three kinds of rule

The kinds — deterministic, unambiguous non-deterministic, ambiguous — and the
overfit defect are defined in the PRD. The checkable rules about rules live on
`ComponentDomainSpec._validation_rules` and `ComponentSourceCode._validation_rules`.

A deterministic rule written as a judged rule is a defect too: it spends a model
call on something a validator can decide exactly, and it makes the outcome
stochastic.

Overfit is the exception the gate cannot catch with confidence. It surfaces when
the user declares that an artifact is correct despite the rule.

The judge returns a typed verdict per rule: `pass` or `fail`, a `reasoning`
field for its argument, and an `evidence` field quoting the data. Keeping the
argument and the quote apart is what makes a contradiction visible instead of
hiding it inside a prose explanation. The judge does not repair the artifact or
propose its correction; a response that does not match the schema is sent back
to the judge with the validation error.

Every step's request, and the judge's, carries an explicit reasoning effort. The
portable "thinking" switch alone is not enough for a gateway route: a dynamic
model id has no profile, so no effort is sent and the provider reasons not at
all. With the effort stated, the provider's thinking is part of the transcript,
and the judge's verdict carries a `reasoning` field holding the argument that is
fed back with a failure.

## Writing a rule

The checkable rules about the rules live on
`ComponentDomainSpec._validation_rules` and
`ComponentSourceCode._validation_rules`; the principles are in the PRD. They are
the model's rules, and the judge is their executor: the authoring prompt renders
them, and the judge's own prompt and code carry none of them. The authoring step
proposes the set as a typed artifact; the judge applies the class rules to that
set; a failure is fed back to the model, which asks the author which reading
they mean and updates it; the framework shows the change; the author confirms.
There is no separate classifier or linter: the model that writes the rules
classifies them — a deterministic rule becomes a code check, an unambiguous one
a judged check — and the author's confirmation is the gate. Code generation
does not start until the set is confirmed.

**Quantifying scope is the author's, not a blanket rule.** A mechanical "say
what happens when there are none" rule was tried and removed: it made the judge
argue about wording instead of evaluating the artifact. The author answers scope
when it matters, and the gate records the answer.

Overfit is the exception the gate cannot catch with confidence — it is handled
at runtime when the author declares an artifact correct despite the rule.

## What a failure means

With the gate in place, a runtime failure has one of three causes:

1. **The definition is ambiguous.** It should have been caught by the gate; if
   it reaches runtime, the author clarifies the rule, and the same artifact is
   re-judged under the clarification. "Every decision has an owner" is the
   example: the author says whether a meeting with no decisions is acceptable.
2. **The user wants something that is not good.** The rule is clear and right;
   the artifact knowingly does not meet it because the user's context differs —
   no next meeting, no action items, headings without the standard sections. The
   author decides: change or scope the rule, or the artifact is not compliant.
3. **The artifact is invalid.** The definition is clear, right and accepted; the
   artifact is malformed, missing a supplied value, or vague where the rule is
   specific. The system repairs or regenerates it; the standard is untouched.

Overfit is a special case of cause 2: the user declares an artifact correct
despite the rule, and the author decides that the rule, not the artifact, was
wrong.

## Decisions

**What is versioned.** The component's requirements — the author's outcomes and
constraints in their own words — are the durable record, with ids and monotonic
versions. Checks, prompts and generated code are specification: each carries the
`requirement_id@version` it serves. An artifact records the requirement versions
it was judged under. "The same object now passes" means re-evaluating that object
against a later version of a requirement.

**What a requirement record holds.** The intent — what the document is for and
what success looks like, in the author's words; the checkable statements; an
exemplar that is good and one that is not, with why; the scope (when it
applies); the owner; and the version. The exemplars are conveyance and test
cases, not rules.

**Who may change what.** Only the author changes a requirement or a rule. The
system owns the implementation: generated code, and the prompt that conveys the
rules. A defect in either is the system's to fix, provided it can name the
requirement it serves.

**Routing a failure.** Name the cause, then act:

1. The user declared a deviation → cause 2: the author decides.
2. The rule is ambiguous → cause 1: the author clarifies, then re-judge.
3. Otherwise → cause 3: the system repairs the artifact.

No one has to prove anything, and a repeat is not a different cause. The
classification is by the evidence in the conversation and the rule's text.

## What we build now

- The **requirement step is the gate**: the model proposes the rule set (a
  deterministic rule as a code check, an unambiguous one as a judged check), the
  framework shows it for confirmation, and the author's acceptance is the gate.
  The rules about rules live on `ComponentDomainSpec._validation_rules` and
  `ComponentSourceCode._validation_rules`; the judge executes them.
- **The feedback loop already exists**: a rejected result is an error message fed
  back to the model, which continues the conversation. The work is the content of
  those messages, not new plumbing.
- **No runtime repetition guard.** A repeated question is a turn the judge
  reads, not a condition the loop detects and corrects. Correcting it at runtime
  hides the model's reasoning failure behind framework prose.
- A component is **released when the author accepts it at the end of the step**.
  There is no separate release action.

The rest — requirement ids and versions, the document store, persisted verdicts,
rework beyond the conversation, and amendments as separate machinery — is
[deferred](../PLAN/standards-and-persistence.md) until the feedback loop proves
insufficient.

The eval of the component-making component still has to show the gate working:
an ambiguous rule is questioned, a deterministic one becomes code, and an overfit
one is caught when the user declares the artifact correct.

## Open questions

None for the minimal path. The deferred design's open questions are in
[document store](document-store.md).
