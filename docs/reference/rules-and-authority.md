# Rules and Authority

A rule is a pass/fail statement about an artifact. This note explains why the
framework must treat a failed rule as a finding about the **standard**, not as
a model error, and who may change the standard.

**Status:** proposal. Scope: one user, present, actively authoring a component.
The absent-author case is out of scope here.

## The problem

The conversation suite surfaced failures that were not framework defects:

- a meeting with no decisions failed "every decision has an owner";
- a required date with no date failed because the only honest value was `TBD`;
- a generated rule said "Location must be Zoom", because one example was Zoom;
- "Note-taker" was rejected as not a role, though it was the user's own word.

Each is a disagreement about the **standard**, not about the conversation, the
protocol, or the model. The current architecture answers every one of them the
same way: retry the model. A retry asks for a *different artifact*. It cannot
fix a rule the facts cannot satisfy, so it either spends the output budget or
produces a fabricated value. Either way the standard has been changed silently,
by a model, with no author.

## Four layers

The word "spec" hides four different things with different owners and lifespans:

| Layer | Example | Durability | Owner |
| --- | --- | --- | --- |
| **Requirement** | "an absent attendee can catch up in two minutes"; "every decision names who made it" | durable, versioned | the author |
| **Check** | the pass/fail predicate that operationalises the requirement | disposable | implementation |
| **Conveyance** | the prompt that tells the model what the requirement is | disposable | implementation |
| **Implementation** | the generated module | disposable, stochastic | generated |

Code generation is **not deterministic from the spec**. A generated module is a
*sample*, not a build product, so the spec cannot guarantee behaviour. The
durable record is the **requirements**, and they are the oracle an artifact is
judged against.

This is the inversion that matters: in ordinary software the code is durable and
the specification rots. Here the generated code is disposable and the
requirements must survive generation.

## Decisions

**What is versioned.** The component's **requirements** — the author's outcomes
and constraints in their own words — are the durable record, with ids and
monotonic versions. Checks, prompts and generated code are specification: each
carries the `requirement_id@version` it serves. An artifact records the
requirement versions it was judged under. "The same object now passes" means
re-evaluating that object against a later version of a requirement.

**Who may change what.** Only the author changes a requirement. The four layers
give the autonomy boundary:

| Layer | System may change it alone? | Condition |
| --- | --- | --- |
| Implementation | yes | the check is faithful; regenerate or repair, then judge again |
| Conveyance | yes, citing the requirement | the requirement is explicit and the prompt omits or misstates it |
| Check | only to repair a **provable contradiction** | the requirement and the check can be shown side by side to disagree; otherwise ask |
| Requirement | never | the author amends it, as a new version |

"Fix the check so this sample passes" is forbidden. "The check contradicts the
requirement" is a bug in the specification and may be fixed autonomously. If the
contradiction cannot be shown from the requirement's own words, the system is
inferring intent, and that is a question for the author.

**Routing a failure.** One diagnosis, then one action:

1. Name the layer from the verdict: which requirement, which check, what
   evidence, what facts the user gave.
2. Check contradicts the requirement, provably → fix the check, re-run, report
   the change against the requirement.
3. Prompt omits or misstates an explicit requirement → fix the prompt, re-run,
   report.
4. A correct artifact is possible from the same facts → repair or regenerate,
   re-run.
5. Otherwise — the requirement is ambiguous, missing, or the fix would change it
   → **ask the author**, with the evidence and the readings, one question.
6. At most **two autonomous attempts per failing requirement** in a run. If it
   still fails, or fails differently, stop and ask. Never loosen a check to pass
   a sample; never change a requirement.

**What the author sees.** An autonomous change is a diff against the check or
prompt, tied to a requirement id, with the re-run result. A question is the
requirement, the evidence, and the readings — not "what should I do?".

**Which validations are structural.** A failure that a corrected object could
fix from the same facts is structural: the object is wrong, and it may be
repaired or regenerated. A failure the facts cannot satisfy is semantic: it is
reported against its requirement, not retried. The two are distinguished by
satisfiability, not by whether a machine or a judge decided it.

## What this implies for the architecture

Not yet implemented. The shape:

- Requirements are modelled, persisted with the component, and versioned; checks,
  prompts and code reference them by `requirement_id@version`.
- Judgment raises a typed **`RequirementFailed`** carrying the verdicts
  (requirement, version, evidence) and the attempted artifact. The caller
  persists both and routes; the exception is the signal, not a second return
  path. Structural validation keeps its own retry path.
- The author has a rework surface: the failing requirement, the evidence, and
  the four-layer choice, with the same artifact re-evaluated against the new
  version.
- The eval of the component-making component includes the loop: a scenario that
  fails a requirement, a routing decision, and the same object passing under the
  amended version. That tests the author's tool, not the generated component.

## Open questions

- How the meta-eval seeds a failure deterministically enough to test each route.
- How the rework surface records an amendment against the component, so the next
  run uses it.
