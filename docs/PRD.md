# PRD — chat-workflow

## Jobs to be done

**When** I need structured, validated data from a person who knows more than I do
(a domain expert, a colleague, a customer),
**I want to** run a guided conversation that asks the right questions in the
right order and refuses to finish with an incomplete answer,
**so that** I get a trustworthy artifact without conducting the interview
myself.

**When** I am the person being interviewed,
**I want** the conversation to sound like an expert thinking with me, not a
form that interrogates me field by field,
**so that** I stay engaged and volunteer the context a rigid form would miss.

## Personas

| Persona | Who | What they need |
|---|---|---|
| **Workflow author** | Developer writing a conversation flow for a domain | Pydantic models that state what a complete answer is; decorators that drive the turn loop; composable steps |
| **End user** | Non-technical domain expert using the flow | Plain language, no workflow jargon, proposals they can react to rather than blank fields |
| **Framework contributor** | Maintainer of the orchestration core | A small, typed surface: config, session, client, decorators, validation mixins |

## User requirements

1. **A flow produces a valid Pydantic value or fails loudly.** Missing or
   contradictory information ends in a named exception, never a half-filled
   object.
2. **Authors compose with ordinary Python.** `@atomic_workflow` for one
   LLM-driven conversation, `@composite_workflow` for a pipeline of them;
   control flow stays in Python (`if`, loops, nested calls).
3. **Validation lives with the data.** Field descriptions, `Validation(...)`
   rules and business-rule validators are attached to the model, so the
   prompt and the check cannot drift apart.
4. **The conversation is a dialogue, not a questionnaire.** The agent
   proposes, asks one thing at a time, and never repeats a question.
5. **Any provider, one configuration.** Provider, model, temperature and
   timeout are read from `config.json`; credentials come from the
   environment. House traffic goes through Cloudflare AI Gateway.
6. **Authors can test the experience, not just the types.** Evals run a
   persona-driven user bot and an LLM judge against the real model.
7. **Blobs and large artifacts are first-class.** A field can materialise to
   a file on disk instead of inflating the JSON result.

## Constraints

- **Cost**: development and CI evals call a real model. Every eval run must
  be bounded (turn limits, timeouts, token reporting) and incremental evals
  are the default on a change.
- **Longevity**: the library is version 0.1 and explicitly unstable. Public
  API changes are allowed; there is no backward-compatibility promise.
- **Backups**: the library holds no user data. Generated workflow artifacts
  and eval transcripts are local files; losing them is acceptable.
- **Monitoring**: no service to monitor. The observable signals are eval
  transcripts, the token/cost report, and named exceptions.
- **Auth**: the library never stores credentials. API keys and the gateway
  token come from the environment; `.env` is untracked.
- **Scale**: one process, one conversation at a time. Concurrency is the
  caller's concern; the framework keeps no global mutable state beyond the
  config singleton.
- **Hosting**: a developer machine or a CI runner. No server, no database.

## Success measures

- A new workflow author can go from `README.md` to a running flow in under
  an hour.
- The eval suite catches a prompt regression (repetition, jargon, fabricated
  values) before a human reads a transcript.
- Every complete flow returns a model that passes its own business rules.
