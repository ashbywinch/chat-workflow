# Writing Documentation Guide

This guide is for anyone writing or updating documentation in this project. It covers the principles and conventions that keep our docs focused, maintainable, and useful.

## Context Efficiency Principle

Every documentation file must only contain information that is relevant to its topic and audience. For each document, be clear about exactly what the (single) topic and intended audience is.

**Before writing a doc, answer these questions:**
- Who is this for? (workflow users, workflow authors, contributors)
- What single question does it answer?
- What information does this audience NOT need?

If a piece of information belongs to a different audience or topic, put it there instead. Don't cross-reference by copying content. Cross-reference by linking.

All documentation must be usable by humans or by AI agents, with the exception of AGENTS.md. Humans and agents must both be easily able to navigate the documentation to find what they need, starting from the entry point of AGENTS.md or README.md (README.md is the entry point for anyone arriving from Github).

### Signs You're Violating Context Efficiency

- A doc has two distinct audiences (e.g., "this section is for contributors, that section is for users")
- A doc covers two unrelated topics
- You're tempted to copy-paste content from another doc
- A reader has to skip large sections to find what they need
- Content is duplicated or concepts are explained twice in broadly the same way, whether within one doc or across several.

## SOLID/DRY Principles for Documentation

### Single Source of Truth

Each piece of information lives in exactly one place. Other docs link to it. They don't repeat it.

**Good:** The contributor guide says "See the testing documentation for details" and links to TESTING.md.

**Bad:** The contributor guide repeats the testing strategy inline.

### One Topic Per File

Each doc file covers one topic for one audience. If you need to cover a subtopic for a different audience, create a separate file and link to it.

### Avoid Redundancy

Before adding content to a doc, check if it already exists elsewhere. If it does, link to it instead of repeating it. If it doesn't, put it in the most logical place and link from other docs.

## Documentation Checklist

Use this to evaluate whether a doc follows Context Efficiency:

- [ ] Single, clearly stated audience
- [ ] Single, clearly stated topic
- [ ] No content that belongs to a different doc
- [ ] No duplicated content from other docs (link instead)
- [ ] Every section is relevant to the stated audience
- [ ] Title and first paragraph make the purpose clear
- [ ] Links to related docs where readers might need them

## How to Update Documentation

1. Identify the audience for your content
2. Find the existing doc for that audience and topic
3. If no doc exists, create one with a clear single purpose
4. Add your content to the right place
5. Update cross-references in other docs (AGENTS.md decision tree, reference tables)
6. Check that you haven't duplicated information that belongs elsewhere
7. Make sure that humans and agents will find your document if they start by reading AGENTS.md or README.md.
