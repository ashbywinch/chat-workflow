# PRD — chat-workflow

This is the single PRD for the product vision and the library that enables it.
It merges the product requirements (R1–R4) with the library requirements
(L1–L12). The conversational surface is specified in [UX](UX.md); the
mechanics and architecture are in [TECHSPEC](TECHSPEC.md); delivery phases are
in [PLAN](PLAN.md).

## Two layers

| Layer | Who | What they do | What they see |
|---|---|---|---|
| **Process owner** | Non-technical expert or business owner | Creates, runs and improves a process in their own language | The conversation and the resulting process; never object models, prompts, checklists or package trees |
| **Workflow author** | Developer using the library | Defines a component: a Pydantic model for the artifact and its workflow steps — object invariants, conversation checklist, and optional procedure | The full library surface: Pydantic models, decorators, checklists, evals |
| **Framework contributor** | Maintainer of the core | Keeps the conversation loop, judge, configuration and evals honest | The core code, tests and eval harness |

Vocabulary: [domain concepts](reference/domain-concepts.md) defines component,
workflow step, workflow spec, and component spec.

The process owner's experience is the product. The author layer exists to make
that experience good and repeatable. The framework layer exists to make the
author layer small and trustworthy.

## Jobs to be done

**When** I own a process and know it well, but not how to specify software,
**I want to** describe it, see a draft, and correct it in my own language,
**so that** I get a process that fits how I actually work without learning a
modelling language.

**When** I am the person the process serves,
**I want** the conversation to behave like an expert thinking with me — not a
form interrogating me field by field,
**so that** I stay engaged and volunteer the context a rigid form would miss.

**When** I am a workflow author,
**I want** a good default checklist and a typed class surface,
**so that** a sensible workflow works out of the box and I only declare what is
specific to my domain.

**When** I am a framework contributor,
**I want** the judge and the end-to-end generation tests to tell me the truth,
**so that** a regression in conversation quality or component generation is
caught immediately.

## Product requirements

### R1 — User-driven improvement

A non-technical person must be able to improve the system according to their
own definition of good, not the vendor's. The system supports this through:

- **Creation:** explicit natural-language elicitation and reflective
  confirmation together. The system asks what matters in the user's language
  and reflects its understanding back for confirmation or correction.
- **Post-conversation reflection:** at the end of a conversation, the system
  reviews improvement signals — for example, the user said “no, not like
  that!” — and offers a concrete improvement to itself for next time. The user
  accepts or rejects it.
- **Planned update sessions:** the user can deliberately open a session to
  update an existing process, rather than only improving it incidentally.

The user never needs to know what an object model, prompt, checklist or package
tree is. The system translates their feedback into the internal changes needed
to improve the process. The mechanism is deferred (see below), but the
capability is a requirement.

Permissions are not part of this system. The requirement is capability, not
authority: a non-technical person can adapt the system. Who that person is, and
what they are allowed to do, is a deployment/tooling concern; downstream tools
may add permissions or choose not to expose the adaptation surface.

### R2 — Thinking partner

The system is a thinking partner. It hides the underlying organisation — the
package tree, object models, prompts and checklists — and helps the user think
intelligently about their process in their own language, using a structured
approach. The result must be that anybody can get a good process without
prompt-engineering or workflow-design expertise.

The system does the structuring. The user can be vague, use examples, or react
emotionally; the system proposes its understanding back for confirmation. The
user is not asked to think like a requirements analyst.

### R3 — Executable operating model

The product is an executable operating model for a business, made of AI
prompts, including an AI prompt designer that can create more prompts and
processes so as to grow the business. Target segments:

1. **Prompt-product sellers** — people selling products based primarily on AI
   prompts, needing a structured way to organise, develop and execute those
   prompts as if they were code.
2. **Business collaborators** — people using a product that collaborates with
   them to create, run and grow a business with a mix of AI and human
   “employees”; primarily solo entrepreneurs at first, but in principle any
   size of organisation.

**Governance.** The product must present a structured approach to constraining
the actions of employees — for example, requiring sign-off from a human
employee or external expert before proceeding. This is an essential part of
AI-in-the-workplace governance. The library does not build a governance
primitive in this phase; the requirement is recorded and deferred.

**Scope.** Software-engineering workflows are a potential use case, not a
current requirement. The current scope is defining, running and improving
processes; broader business planning is not part of the current requirements.

### R4 — Information ownership

Important business information must have clear ownership by a specific expert
role — not just capture by automated functions. The owner is responsible for
maintaining, interpreting and using that information for business decisions.
New business functions must specify an owner; the system makes ownership
explicit rather than ad hoc.

Ownership is a property of the **package tree**, not of each class. Every class
lives in a package; packages nest to form a tree; each node has an owner, like
an org chart. A class is governed by the owner of its package node. How the
owner is recorded is an open question (deferred).

## The definition of excellence

The **workflow author** defines what good looks like for the objects of their
class and for the conversations that create and update those objects. In the
evals, the project authors are the workflow authors; the rubric is ours, not a
vendor imposition.

The library supplies a **good default** so that an author who writes a method
without a rubric still gets sensible behaviour. The default set applies to
every method; a method can add its own rules with
`conversation_validation_rules`.

### Three separate things

| Artifact | What it is | Where it lives | Enforced by |
|---|---|---|---|
| **Object invariants** | Object quality. Must be true after construction and after a workflow step modifies the object. | Pydantic fields, `Validation(...)`, model validators | Pydantic validation |
| **Procedure** | How to have the conversation to build the object. A skill or business-process description. | Method docstring. Optional; a good default is the goal. | Prompt construction |
| **Conversation checklist** | What a good conversation looks like. | The workflow step (`conversation_validation_rules`), with the framework default | The eval judge |

The object invariants are not judged by the conversation judge. The checklist is
not object quality. The procedure is not the checklist; the docstring describes
how to do the work, the checklist evaluates whether the conversation was good.

### The epistemic principle

The two apparently competing ideas — “propose one thing at a time” and “show the
complete picture” — are one principle:

- **Unknown details:** when the assistant proposes something the user has not
  said or agreed, it introduces one such hypothesis at a time and asks for
  confirmation. It never infodumps a batch of guesses.
- **Agreed details:** once the user has said or agreed the material, the
  assistant may synthesise it into a complete draft and present it for
  approval. This is the “circulate the draft after the meeting” move; the
  draft is labelled as a draft, not asserted as fact.

Unknown → one at a time. Agreed → synthesise into a draft for approval.

### Default conversation checklist

The framework defines this default once in
`chat_workflow/conversation_rules.py` and applies it to every method. A
method's `conversation_validation_rules` adds its own rules to the floor;
declaring nothing gives the default.

1. **No hypothesis dumps.** When the assistant proposes something the user has
   not said or agreed, it introduces one such hypothesis at a time and asks
   for confirmation. It never presents a batch of guessed details as if they
   were established.
2. **Complete draft only from agreed facts.** The assistant may present the
   complete draft in one message for approval only when every element is
   grounded in what the user said or agreed. It labels the result as a draft
   for approval rather than asserting it as fact.
3. **No repetition.** It does not ask again for information the user has
   already given.
4. **Incorporates context.** It builds on what the user has said rather than
   restarting or ignoring it.
5. **Uses the user's language.** It does not introduce framework or model
   jargon the user has not used.
6. **No schema interrogation.** It does not ask about a field just because it
   exists in the schema. If domain knowledge suggests the field does not apply
   but the user has not said so, it confirms that assumption in the user's
   language — for example, *“So we start with your raw meeting notes, and
   enhance them from there?”* Every field must be something the user can know
   and confirm; implementation facts such as “who calls this” are never fields.
7. **Thinking partner, not a form.** It proposes concrete ideas grounded in
   what the user said and asks for a reaction, rather than interrogating the
   user field by field. A follow-up question on the point under discussion is
   fine; the failure is an unrelated open question that ignores the user's
   last message. When the user rejects a proposal or asks for the assistant's
   opinion, it responds with one concrete alternative proposal or
   recommendation.
8. **Explains purpose.** It explains what the conversation is trying to
   achieve and why.
9. **Listens first.** It asks open-ended questions and absorbs what the user
   says before proposing structure.

### Conversation-level rules

**Warm open** applies only to the first method in a conversation. The first
assistant response must orient the user and explain what the conversation is
for before substantive questioning. The conversation layer — not the individual
workflow class — determines which response is first. A workflow class must be
reusable mid-conversation without behaving as if it were the opening. The
mechanism is a design decision (deferred).

### Author additions

An author can add method-specific rules to the checklist. Some rules are too
specific for the shared default — for example, “do not take over the user's
task and produce the artifact for them” applies to design conversations, not to
a component whose job is to produce the artifact. Such rules live on the
methods that need them, not in the shared rule module.

## Library requirements

### L1 — Valid typed output or explicit failure

A flow produces a valid Pydantic value or fails with a named exception. It never
returns a half-filled object. Missing or contradictory information ends in a
failure the caller can handle.

### L2 — Composition in ordinary Python

Authors compose workflows with `@atomic_workflow_step` and `@composite_workflow_step`.
Control flow stays in Python: `if`, loops, nested calls. Deterministic
sequencing is code, not prompt text.

### L3 — Object invariants live with the data

Field descriptions, `Validation(...)` rules and business-rule validators are
attached to the model. All invariants must hold after construction and after a
workflow step modifies the object. Workflow steps that return a newly
constructed Pydantic instance are validated by construction; a method that
mutates an existing instance must return a validated instance (construct a new
one or revalidate before returning). The current workflow path returns
constructed instances, so the normal path is covered; this requirement closes
the in-place-mutation gap.

### L4 — Conversation checklist on the method

The checklist is declared on the workflow step with
`conversation_validation_rules`, a list of plain criterion strings. The
framework applies its default set to every method; a method's declarations add
to it. The judge consumes the method's checklist.

### L5 — Procedure is optional

The method docstring is the skill/process description. It is optional; a good
default is the goal. Deterministic control flow is always in code, never in the
procedure text. A blank docstring must not mean “no instructions”; it means
“use the default.”

### L6 — Conversation-level rules are not class-level

A rule that depends on the conversation's position — such as warm open applying
only to the first method — belongs to the conversation layer. The domain class
must not know it is first, and the same class must work mid-conversation.

### L7 — Any provider, one configuration

Provider, model, temperature and timeout come from `config.json`; credentials
come from the environment. House traffic goes through Cloudflare AI Gateway.

### L8 — Evaluation is layered and honest

- **Deterministic unit tests** are the gate; no LLM calls.
- **One-shot evals** use a fixed input and a real model, and assert the typed
  result. No user bot, no judge.
- **Multi-turn judge evals** are the point of conversation-quality testing. The
  judge applies the class checklist and reports per-rule verdicts.
- **End-to-end generation evals** are critical: generate a component, compile
  it, import it, run it and validate the result. If any stage fails, the eval
  fails at that stage, immediately, with the generated code preserved.
- The judge never runs before the workflow produced a valid typed result. A
  workflow failure is never reported as a conversation-quality failure.

### L9 — Blobs and large artifacts are first-class

A field can materialise to a file on disk instead of inflating the JSON result.

### L10 — No auth, no server, no database in the library

The library is a single process, one conversation at a time. It has no auth,
no permissions, no server and no database. Multi-user operation, auth,
persistence and hosting belong to the product layer.

### L11 — Ownership is a package-tree property

The library models ownership on the package tree, not on individual classes.
The recording mechanism is deferred.

### L12 — The default must be good

A method with no checklist and no procedure should still produce a good
conversation. The default checklist and the default procedure are therefore
part of the product quality bar, not an afterthought.

## Evaluation model

Build-time evaluation belongs to the author/framework layer. Post-release
user-driven evaluation belongs to the process owner (R1); the two must not be
confused.

| Layer | Who runs it | What it answers |
|---|---|---|
| Unit tests | Author/contributor | Does the code obey its contract? |
| One-shot evals | Author/contributor | Does the prompt produce a valid typed result from a known input? |
| Judge evals | Author/contributor | Did the conversation meet the author's checklist? |
| End-to-end generation | Author/contributor | Can the system generate a component that compiles, imports and runs? |
| Post-release reflection | Process owner | What should improve next time, in the user's own judgement? |

The judge is the executor of the author's checklist. It is not a separate
opinion and it is not optional for conversation quality.

## Constraints

- **Cost:** development and CI evals call a real model. Every run is bounded
  (turn limits, timeouts, token reporting); incremental evals are the local
  default. The full eval suite is a scheduled/manual run, not a per-PR gate.
- **Longevity:** the library is version 0.1 and explicitly unstable. Public API
  changes are allowed; there is no backward-compatibility promise.
- **Backups:** the library holds no user data. Generated artifacts and eval
  transcripts are local files; losing them is acceptable. Persistence is a
  product-layer concern.
- **Monitoring:** no service to monitor. The signals are eval transcripts, the
  token/cost report, and named exceptions.
- **Auth:** none in the library. The product layer may add it; downstream tools
  may add permissions.
- **Scale:** one process, one conversation at a time. Concurrency, multi-user
  operation and hosting are the product layer's concern.
- **Hosting:** a developer machine or a CI runner. No server, no database.
- **Governance:** human/expert sign-off is a product requirement, deferred; no
  library primitive in this phase.
- **Scope:** process definition, execution and improvement. Software-engineering
  workflows are a potential use case, not a current requirement. Broader
  business planning is not part of the current requirements.

## Success measures

- A new workflow author can go from the README to a running flow in under an
  hour.
- A class with the default checklist produces an acceptable conversation
  without the author writing a rubric.
- The eval suite catches a prompt regression — repetition, jargon, fabrication,
  hypothesis dumps — before a human reads a transcript.
- End-to-end generation fails immediately when a component cannot be generated,
  compiled, imported or run.
- A non-technical process owner can improve a process through natural-language
  feedback without seeing the internals.
- Every complete flow returns a model that satisfies its own invariants.

## Non-goals

- Auth, permissions, multi-user operation, a server or a database in the
  library.
- A general business-planning tool.
- A fixed vendor-defined definition of good that the author cannot replace.
- Deterministic control flow expressed in prompt text.
- Backward compatibility for the 0.1 library.
- A full governance/sign-off system in this phase.

## Deferred decisions

| Decision | Direction / notes |
|---|---|
| Governance and sign-off primitive | Product requirement; no library primitive in this phase. |
| Org-chart visibility | Probably show the domain role in the user's language, not the technical package tree; decide when building the product. |
| How ownership is recorded | Open; the package tree is the model, the recording mechanism is not chosen. |
| Default procedure when the docstring is blank | Deferred. Deterministic control flow belongs in code; the docstring is a skill description. |
| Warm-open wrapper design | Deferred. The conversation layer owns “first”; the domain class must not know. |
| Non-technical adaptation mechanism | The capability is required (R1); the mechanism for translating feedback into internal changes is deferred. |
| Judge mechanics | Temperature, model, call shape and report format belong in the TECHSPEC. |
| Product-layer auth, persistence and hosting | Out of scope for the library; belongs to the product layer. |
