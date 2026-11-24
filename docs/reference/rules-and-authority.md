# Rules and Authority

A rule is a statement of what "good" means for an artifact. This note explains
the three kinds of rule, how each is evaluated, what a failure means, and who
may change the standard.

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

Ambiguous and non-deterministic are different things. A rule can be impossible
to check mechanically and still be perfectly clear; that is the kind of rule this
system exists to evaluate.

| Kind | Example | Evaluated by |
| --- | --- | --- |
| **Deterministic** | "any action has a date" | code — a field constraint or validator |
| **Unambiguous, non-deterministic** | "the minutes capture the actions agreed, as per the notes" | the judge |
| **Ambiguous** | "the description must be specific enough" | nobody — it is a defect |

A deterministic rule written as a judged rule is a defect too: it spends a model
call on something a validator can decide exactly, and it makes the outcome
stochastic.

A fourth defect is **overfit**: a rule that encodes one example ("Location must
be Zoom"). It is unambiguous, so it cannot be detected from its text with
certainty. It surfaces when the user declares that an artifact is correct
despite the rule.

## Rules about rules

Rules are authored by the author and checked against a small set of
meta-rules at authoring time:

1. **One requirement per rule.** A compound rule hides one failure behind the
   other.
2. **Quantify the scope.** Say whether at least one must exist, or whether only
   the items that exist are constrained. "Every decision has an owner" does not
   say whether a meeting with no decisions is acceptable.
3. **No undefined predicates.** "Specific", "clear", "appropriate", "good" name
   nothing checkable; replace them with the parts that make a thing so.
4. **No example values as constraints.** An example belongs in the exemplar,
   not in the rule.
5. **If it can be checked mechanically, it must be checked in code**, not judged.
6. **Cite the user's words**, so the author recognises their own intent.

The meta-rules are **the validation rules of the requirement set itself**. The
authoring step proposes the set as a typed artifact; the framework shows it to
the author for confirmation; a rule that fails a meta-rule is fed back to the
model as a correction, and the model continues the conversation and asks the
author to disambiguate. There is no separate classifier and no linter: the model
that writes the rules classifies them — a deterministic rule becomes a code
check, an unambiguous one a judged check — and the author confirms the result.
The gate is that loop: code generation does not start until the set is confirmed
and the meta-rules pass.

The authoring prompt carries one explicit hint: if a rule is ambiguous, ask the
author which reading they mean before finishing; never choose one for them.
Overfit is the exception the gate cannot catch with confidence — it is handled at
runtime when the author declares an artifact correct despite the rule.

## What a failure means

With the gate in place, a runtime failure has one of three causes:

1. **The definition is ambiguous.** It should have been caught by the gate; if
   it reaches runtime, the author clarifies the rule, and the same artifact is
   re-judged under the clarification.
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
  framework shows it for confirmation, and a meta-rule failure is fed back so the
  model asks the author to disambiguate. No separate classifier.
- **The feedback loop already exists**: a rejected result is an error message fed
  back to the model, which continues the conversation. The work is the content of
  those messages, not new plumbing.
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
