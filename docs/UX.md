# UX — chat-workflow

The user experience is the conversation itself. This spec covers the
conversational surface the framework produces; it is binding on workflow
prompts.

## Principles

- **A thinking partner, not a form.** Propose a concrete idea and ask for a
  reaction. Blank-field questioning is the failure mode.
- **One thing at a time.** One question per turn; never stack unrelated
  asks.
- **Plain language.** No workflow jargon in user-facing text. Say "steps"
  or "parts", not "stages", "phases", "process structure", "deliverable",
  "consumer" or "stakeholder" unless the user used the term first.
- **Never repeat.** If the user already answered, use the answer. A repeated
  question is a defect, not a nudge.
- **Separate inference from fact.** Proposed values are tentatively
  asserted ("I'm guessing X — is that right?"). Never silently fabricate a
  value.
- **Honest progress.** The user can tell what has been settled and what is
  still open without reading a spec.

## Conversation shape

1. **Open with the problem**, not the schema.
2. **Reflect back** what was heard in the user's language.
3. **Propose the shape** of the result, then ask for confirmation.
4. **Fill gaps one at a time**, suggesting an option when the user is stuck.
5. **Confirm the final artifact** in plain language before returning.

## Failure and recovery

- If the user does not know an answer, offer one plausible option and mark
  it as a suggestion.
- If the turn limit is reached, fail with a message that names the limit and
  what remains open — never return a partial object.
- If validation fails, say which rule failed in the user's language and ask
  for the specific correction.

## Accessibility of language

- Short sentences; no nested clauses.
- Define a domain term the first time it appears.
- Avoid metaphor that implies technical structure ("pipeline", "node",
  "schema").

## What good looks like

A user with no workflow background describes their problem in two or three
messages, reacts to one proposal, answers a handful of targeted questions,
and receives an artifact they recognise as their own thinking. They never
had to learn the framework's vocabulary.
