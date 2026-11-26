# Domain Concepts

This document defines the core vocabulary of the chat-workflow library and maps it to equivalent concepts in the broader LLM ecosystem. Understanding these terms helps newcomers leverage existing domain knowledge when reading or contributing to the codebase.

## Workflow

**What it is:** The whole: the components that produce the deliverable, the
workflow steps they own, and the order they run in. A workflow may be one
component with a few steps, or a chain of components that each own part of the
work.

**Not:** A workflow step, and not the workflow spec. A workflow step forms part
of a workflow; a workflow is never a kind of step. The spec is the design of
the workflow, not the workflow itself.

**In the code:** the `WorkflowSpec` model and the components it lists.

## Workflow Step

**What it is:** A piece of work that forms part of a workflow. A workflow step
is **atomic** when it runs one LLM conversation that produces or changes an
artifact; it is **composite** when it runs other workflow steps, including
steps owned by other components.

**Broadly known as:** an *activity* or *sub-process* (BPMN), a *step*
(Temporal, Airflow), a *task* in most workflow engines.

**In the code:** a method decorated with `@atomic_workflow_step` or
`@composite_workflow_step`.

## Atomic Workflow Step

**What it is:** The smallest unit of LLM-driven conversation that produces a validated Pydantic object. An atomic workflow step runs a multi-turn loop (user ↔ LLM) until the LLM signals success with a valid result or failure, or the turn limit is reached when the step declares one.

**Broadly known as:** In other frameworks this concept appears as a *structured output agent loop*, a *function-calling agent*, a *tool-use loop*, or a *run* (OpenAI SDK). Vercel's AI SDK calls it a multi-step *generation* with `stopWhen`. LangGraph implements it as a *StateGraph* that loops until a condition is met. LlamaIndex has *FunctionAgent* for the same pattern.

**In the code:** `AtomicWorkflow` (class), `@atomic_workflow_step` (decorator), `process_turn()` (method that drives one user → LLM cycle).

## Composite Workflow Step

**What it is:** A workflow step that composes multiple atomic workflow steps
and other composite workflow steps into a larger piece of work. It does not
drive its own LLM loop — it passes a `Session` object to child steps and
orchestrates their execution.

**Broadly known as:** *Multi-agent workflow* (LlamaIndex), *workflow graph* (LangGraph), *pipeline* or *orchestration*. The pattern of composing reusable steps into a larger process is universal in workflow systems (Amazon States Language, Temporal, Airflow).

**In the code:** `@composite_workflow_step` (decorator), `Session.run()` (drives an atomic workflow step within a composite), `generate_reviewed_criteria` (example).

## Component

**What it is:** A class that owns exactly one business artifact — the set of
data members of the Pydantic model it defines — and the workflow steps that
create or update it. The class may have several steps on that one artifact
(`create`, `update`, and other operations its lifecycle needs), and it may be
responsible for the whole deliverable by delegating to other components that
own their own artifacts. Types it uses or returns from steps are dependencies,
not its artifact.

When a component delegates, its artifact is still the finished thing — the
plan, the report, the minutes — assembled from what the delegated components
produce. The delegated components own subsets of it (a section) or temporary
artifacts used along the way (raw notes). The coordinator's artifact is not
"coordination"; coordination is how it produces its artifact.

**Not:** A workflow, a component spec, or the workflow spec. A business model is
an example of a component, not something a component owns.

**In the code:** `Component` (generated record), `ComponentSourceCode`
(generated source), a generated class such as `MinutesDraft`. Examples:
`MinutesDraft`, `Invoice`, `BusinessModel`.

## Workflow Spec

**What it is:** The specification of a top-level workflow (a business process):
its name, diagram, inputs, outputs, and component specs. It is a design
artifact, not a runtime object. The spec names every component the process
needs, including indirect dependencies, because it must describe the whole
thing.

**Not:** A workflow step, and not the owner of the components. A
workflow spec has component specs; it does not have components.

**In the code:** the `WorkflowSpec` model.

## Component Spec

**What it is:** The design-time description of one component: its name,
purpose, scope, required inputs, and type. It is what a workflow spec lists so
the process can be implemented end to end.

**Not:** The generated component class. The spec describes what the component
must be; the generated component is the code that satisfies it.

**In the code:** `ComponentSpec`.

## Has, Uses, Needs

“Workflow” alone means the whole workflow — its components and their workflow
steps — not a workflow step and not the workflow spec. A workflow step is
always named as such.

| Statement | True? | Why |
|---|---|---|
| A workflow has components. | ✓ | The components are what the workflow is made of. |
| A component has workflow steps. | ✓ | The steps produce and change the component's artifact. |
| A workflow step forms part of a workflow. | ✓ | Through the component that owns it. |
| A workflow step has a component, parameters, and a return type. | ✓ | The bound class, the signature, and the return annotation. |
| A workflow is a workflow step. | ✗ | A workflow step forms part of a workflow. |
| A workflow spec has component specs. | ✓ | The specs are data in the spec; a dependency list, not ownership of the components. |
| A workflow spec has components. | ✗ | It has component specs; the components are separate code artifacts. |
| A component has dependencies. | ✓ | It can depend on other components; it does not own them. |
| A component represents a business artifact. | ✓ | The artifact is the model's data members; its steps create or update it. |
| A component has exactly one artifact. | ✓ | The artifact is defined as the set of data members of its model. |
| A component has exactly one workflow step. | ✗ | It may have as many steps on its one artifact as the lifecycle needs. |
| A workflow needs a coordinator component. | ✗ | Coordination is an optional pattern; a pipeline of direct handoffs is valid. |

Example: a process for making a business plan needs a vision/mission definition
and a strategy, but the process does not *have* the vision or the strategy. The
workflow spec lists the component specs that will define them.

## Turn

**What it is:** A single LLM round trip: the framework sends a user message (plus accumulated history) to the LLM, receives a response, and acts on it. If the response is "continue", another turn begins.

**Broadly known as:** *Turn* is the standard term (OpenAI SDK, Agent Client Protocol, Microsoft Agent Framework). Vercel's AI SDK calls it a *step*. LangGraph calls it a *super-step*.

**In the code:** `turn_count` (field), `TurnLimitExceededError`, `TurnResult`, `process_turn()`. The concept is already correctly named.

## Session

**What it is:** One conversation. A session wraps IO (how to talk to the user), its own state log (the messages and audit events of that conversation), and config (provider, model, timeout). A session may create child sessions: a meeting inside a conversation is the same concept at a deeper level. A child starts with clean context and receives its brief as parameters; it does not inherit the parent's messages. The tree of sessions is the whole the user saw.

**Broadly known as:** *Session* is widely used (OpenAI SDK, Microsoft Agent Framework, Agent Client Protocol). It represents durable, multi-turn state that survives individual LLM calls. Other frameworks call the same concept *runtime context* or *invocation context*; the recursive child session is their *sub-session* or *nested run*.

**Not:** A second message store. The conversation stays in `SessionLog.messages`; `Session` is the context that owns it. A child session is not a different class.

**In the code:** `Session` (class), `SessionLog` (that session's transcript), `Session.spawn()` (a clean child), injected as the `session` parameter in workflow steps.

## Session Component

**What it is:** The behaviour that runs a session: it greets once, runs a step, and signs off. It is stateless and wraps any session at any depth. It owns the conversation's opening and closing so an individual workflow step never has to.

**Broadly known as:** the *agent harness* or *conversation shell* around a step. Other frameworks put the greeting and closing in a runtime layer rather than in each task.

**In the code:** `SessionComponent` (class), `greet`, `converse(step)`, `sign_off`.

## Agent Response

**What it is:** The structured object the LLM returns after one turn. It carries an `intent` (what the framework should do next) and optionally a `message` for the user or a `result` (the validated Pydantic object).

**Broadly known as:** *Structured output* (Vercel AI SDK), *tool call* / *function call* (OpenAI API), *completion* with a *finish reason*. The `intent` field aligns with what other frameworks call *stop reason*, *finish reason*, or *routing decision*.

**In the code:** `AgentResponse` (model), `AgentIntent` (enum: `CONTINUE`, `SUCCESS`, `FAILURE`).

## Turn Result

**What it is:** The framework's internal outcome after processing one turn — wraps the LLM's response into a standard shape (`is_complete`, `message`, `result`) used by the turn loop.

**Broadly known as:** This is plumbing analogous to a *step result* or *node output* in other frameworks. It's the return value of one iteration of the agent loop.

**In the code:** `TurnResult` (model), with factory methods `continuing()`, `success()`, `failure()`.

## How These Fit Together

```
Session (one conversation; may spawn child sessions)
  ├── SessionComponent (greet, run, sign off)
  └── Workflow
        └── Component (one artifact)
              ├── Atomic Workflow Step       ← has its own turn loop
              │     ├── Turn 1 (user → LLM → AgentResponse)
              │     ├── Turn 2 (user → LLM → AgentResponse)
              │     └── ... → returns TurnResult
              └── Composite Workflow Step    ← runs other steps
                    └── (steps on this or other components)
```

The outer container is a **Session** (one conversation). A **SessionComponent** runs it: it greets, runs the workflow, and signs off. Inside it, a **Workflow** is made of **Components**; each component owns one artifact and the workflow steps that produce and change it. An **Atomic Workflow Step** runs a **turn** loop until the LLM returns an **Agent Response** with `intent=SUCCESS` or `intent=FAILURE`. A **Composite Workflow Step** runs other steps. Each response is wrapped in a **Turn Result** and recorded in the session's log. A step that routes to a meeting calls `Session.spawn()` and runs the child session's step through the same `SessionComponent`.

## Blob Field

**What it is:** A Pydantic field annotated with `Blob(extension)` whose content is automatically materialized to a file on disk after a workflow produces a result. The blob's serialized value is the file path rather than the content itself, enabling clean handling of large text outputs (markdown, code, structured data).

**Broadly known as:** *File attachment* or *sidecar file* in other frameworks. Similar to how Gale-Shapley `Blob`s work in semantic-kernel or how an MCP server returns `BlobResource` contents. The core idea is an in-memory value that maps to a real file.

**In the code:** `Blob` (annotation, in `chat_workflow/annotations.py`), `BlobSyncMixin` (mixin, in `chat_workflow/mixins.py`).

## Validation Rule

**What it is:** A natural-language business rule attached to a field via `Annotated[str, Validation("rule text")]`. The rule is enforced by the LLM during validation — rather than a hard-coded constraint in Python, the validator checks whether the field's value satisfies the described rule. Multiple validation rules can be stacked on a single field.

**Broadly known as:** *Semantic validation* or *LLM-as-judge*. Unlike traditional schema validation (regex, min/max, type checks), a validation rule captures subjective or qualitative constraints that are better evaluated through language understanding than arithmetic.

**In the code:** `Validation` (annotation, in `chat_workflow/annotations.py`), `LLMValidated` (mixin, in `chat_workflow/mixins.py`).

## Gap Resolution Loop

**What it is:** A looping pattern inside a `@composite_workflow_step` function where an atomic workflow step identifies gaps or issues in a result, a follow-up refinement step addresses them, and the cycle repeats until all gaps are resolved. Each iteration narrows the delta between the current state and the desired outcome.

**Broadly known as:** *Iterative refinement*, *critique-revise loop*, *reflection agent pattern*. LangGraph implements this as a *feedback loop* between nodes. Vercel's AI SDK supports it through chained generations. The key idea is that a second pass (often a different prompt or model) finds flaws the first pass missed.

**In the code:** The pattern is visible in `generate_reviewed_criteria` (in `workflows/evaluation_criteria/flows.py`), where a refinement step loops over gap analysis results until the reviewer is satisfied.