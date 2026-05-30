# chat-workflow-prototype Migration Guide

## TL;DR

This document explains how concepts from the `chat-workflow-prototype` repository (an `.md`-based AI-executable operating model) map to the `chat-workflow` Python library. If you worked with the prototype's playbook-driven architecture, this guide shows you the equivalent concepts and helps you understand the shift from markdown-defined processes to Python-defined workflows.

The prototype was a working proof of concept. It proved that an AI agent could execute business processes defined entirely in structured markdown. The `chat-workflow` library takes the same idea and implements it in deterministic Python code: Pydantic models replace Document Quality Standards (DQS), decorated Python functions replace playbooks, and package structure replaces component folders.

## What Was the Prototype?

The `chat-workflow-prototype` repository (at https://github.com/ashbywinch/chat-workflow-prototype) was an experiment in "Business as Code." It contained zero Python files. It was a directory of markdown files organized as an AI-executable business operating model.

The core insight came from the bootloader:

> "You are a programmer/computer, not a technical writer."

An AI agent reading the prototype would interpret business documents as executable instructions, not as passive documentation. A DQS was a data structure. A playbook was a function. A business component was a class. The entire system was shaped like a Fortune 500 business, with strategic value streams, operational domain objects, coordination components, and a single main entry point.

The prototype was the insight. The `chat-workflow` library is the implementation.

## Key Prototype Concepts

These are the concepts the prototype actually defined. They live entirely in markdown files and are executed by an AI agent that reads them.

### 1. AGENT.md (Bootloader)

The entry point. Located at the root of the repository, it sets the paradigm that the AI is a computer executing code, not a technical writer writing documentation. It defines the "operating model / object oriented code" analogy:

- DQS documents are like data structure definitions
- Playbooks are like functions
- Business Components are like classes
- Artifacts are like objects

It also establishes the **DQS-First** approach: before creating or modifying any artifact, always find and read the relevant DQS first.

### 2. DQS (Document Quality Standard)

Markdown documents that define "what good looks like" for a business artifact. A DQS is analogous to a type definition or schema. It is not a template to fill in. It is a standard to conform to.

A typical DQS has these sections:

- **Title**: "[Subject] Document Quality Standard"
- **Target Document Definition**: What the artifact is
- **Purpose Clarity**: What the standard achieves
- **Structure**: Numbered sections with descriptions and quality criteria
- **What Good Looks Like**: Extra evaluation criteria
- **Validation Checklist**: Binary pass/fail checks
- **Common Anti-Patterns**: What to avoid
- **Related Standards**: Cross-references

The prototype had DQS documents for everything: business components, workflows, engagement playbooks, and even DQS documents themselves.

### 3. Playbook

Markdown documents that define executable business processes. Each playbook is analogous to a function. It has:

- **Purpose**: What the playbook accomplishes
- **Expert Role**: Who drives the conversation (e.g., "Business Architecture Specialist", "System Architecture Specialist")
- **Parameters**: Required and optional inputs, each referencing specific DQS document types
- **Return Types**: Explicit outputs, each referencing DQS conformance
- **Steps**: Numbered sequential instructions
- **Outcome**: What is produced
- **Success Metrics**: Objective quality criteria
- **Related Documents**: Cross-references

Playbooks can call other playbooks, just like functions call other functions. For example, the Create Business Component Playbook calls Create Engagement Playbook Playbook as a sub-step, making it composite.

### 4. Business Component

A folder that groups together a DQS and its related playbooks. The fundamental unit of organization. Components mirror a Fortune 500 role hierarchy with two categories:

- **Value Stream Components**: Strategic areas with KPIs and budgets
- **Domain Object Components**: Operational units that generate artifacts on demand

Each component has a single DQS (defining its artifact type) and a set of playbooks (defining how to create, modify, or use those artifacts).

### 5. Composability

Playbooks can call other playbooks. This is explicit in the prototype's architecture standards:

> "Playbooks can call other playbooks, just like functions calling other functions."

A component should not concern itself with how it will be called. The calling playbook handles orchestration. This is the same principle as dependency injection.

### 6. DQS-First Approach

The single most important operational rule in the prototype: always find and read the relevant DQS before creating or modifying any artifact. Check compliance throughout. If no DQS exists, ask the user if one should be created.

### 7. Expert Role

Each playbook assigns a single expert role responsible for execution. Examples include "Business Architecture Specialist", "System Architecture Specialist", "PA". This role drives the conversation. Only one role is permitted per playbook due to single-thread execution constraints.

## Mapping to chat-workflow

| Prototype Concept | chat-workflow Equivalent |
|---|---|
| AGENT.md (bootloader) | AGENTS.md + workflow discovery in `chat_workflow_cli/cli.py` |
| DQS Document (quality standard) | Pydantic `BaseModel` -- defines structure, types, validation |
| Playbook (executable process) | Python function with `@atomic_workflow` or `@composite_workflow` decorator |
| Business Component (folder with DQS + Playbooks) | Python package under `workflows/` with models + workflow functions |
| Playbook Parameters (required/optional) | Function parameters + `Annotated` descriptions |
| Return Types | Pydantic return type annotation on the function |
| Expert Role | System prompt in the workflow function's docstring |
| Steps (numbered process) | Atomic workflow orchestrated by composite workflow |
| DQS Compliance Checklist | Pydantic validators (`min_length`, `ge`, `le`, `@model_validator`) |
| Related Documents section | Cross-references in `docs/` |
| Component Index File | `__init__.py` that exports workflow functions for CLI discovery |
| Composability (playbooks calling playbooks) | Composite workflows calling atomic workflows |

### What the Mapping Means in Practice

**DQS to Pydantic Model**: A DQS defines what fields an artifact has, what types they are, what valid ranges exist, and what business rules apply. A Pydantic `BaseModel` does the same thing, but with deterministic enforcement. The `description` parameter in `Field()` carries the "what good looks like" guidance that the prototype encoded in DQS sections. The `ge`, `le`, `min_length`, and `@model_validator` checks replace the validation checklist.

**Playbook to Workflow Function**: A playbook has a purpose, parameters, return types, and steps. A Python function with `@atomic_workflow` has the same structure: the docstring is the purpose (and the system prompt), the function signature is the parameter list, the return type annotation is the return type, and the decorator handles the step-by-step conversation loop.

**Business Component to Python Package**: A component folder contains a DQS and related playbooks. A Python package under `workflows/` contains a Pydantic model file and workflow function files, exported through `__init__.py`. The CLI discovers packages automatically.

**Parameters and Return Types**: In the prototype, parameters reference specific DQS document types (e.g., "Business Idea document conforming to Business Idea DQS"). In chat-workflow, parameters have Python types and `Annotated` descriptions. Return types are Pydantic models. The intent is the same: typed inputs produce validated outputs.

## Patterns Worth Carrying Forward

The prototype contained several design principles that translate well to the chat-workflow library, even though the implementation mechanism is different.

### DQS-First: Read the Standard Before Creating

In the prototype, you never create an artifact without first reading the DQS. In chat-workflow, this means understanding a model's validation rules before writing workflow prompts. Know what constraints the Pydantic model enforces, and write prompts that guide the LLM to produce data that satisfies those constraints.

### Playbook as Function: Clear Parameters and Return Types

Every playbook in the prototype declares its parameters and return types explicitly. This maps directly to typed Python functions with Pydantic I/O. The pattern is the same: you know what goes in and what comes out before you start writing the body.

### Composability via Orchestration

Playbooks calling playbooks is the same pattern as composite workflows calling atomic workflows. The prototype showed that this pattern works for business processes. Chat-workflow makes it explicit in code with two decorator types: one for atomic conversations, one for orchestration logic.

### Expert Role as Prompt

Each playbook assigns an expert role. In chat-workflow, this goes in the atomic workflow's docstring. It tells the LLM how to behave, what perspective to take, and what kind of questions to ask. This is the simplest and most effective way to shape conversation quality.

### AI as Computer, Not Technical Writer

This is the most important insight from the prototype. The AGENT.md bootloader says:

> "You are a programmer/computer, not a technical writer."

This mindset informs how we write prompts: be precise, treat the LLM as a reasoning engine, not a creative writer. The prototype proved that AI agents can execute structured processes reliably when the instructions are clear, typed, and composable. Chat-workflow takes this further by making validation deterministic through Python's type system.

### Dual-Placement Learning

The prototype's architecture standards describe a pattern where process learnings go in playbooks (for prevention) and validation standards go in DQS documents (for detection). This maps to chat-workflow as: prompt guidance goes in workflow function docstrings (preventing mistakes during generation) and validation rules go in Pydantic models (catching mistakes after generation).

### Compilable Format

The Engagement Playbook DQS includes a section on "Compilable Format Requirements" that anticipates automatic compilation of playbooks to Python functions. The chat-workflow library fulfills this vision: playbooks are now Python functions, DQS documents are Pydantic models, and the compiler is the Python runtime.

## Recap: The Big Shift

| Aspect | Prototype (chat-workflow-prototype) | Current (chat-workflow) |
|---|---|---|
| Implementation | Markdown files (.md) | Python code (.py) |
| Structure definition | DQS document (sections, checklists) | Pydantic BaseModel (fields, types, validators) |
| Process definition | Playbook document (numbered steps) | Decorated Python function (atomic or composite) |
| Unit of organization | Business Component folder | Python package under `workflows/` |
| Validation | AI agent interprets DQS checklist | Python enforces types + Pydantic validates |
| Conversation flow | AI agent reads playbook steps, executes them | `@atomic_workflow` decorator manages LLM conversation |
| Composability | Playbooks call other playbooks via markdown references | Composite workflows call atomic workflows via function calls |
| CLI discovery | None (agent reads AGENT.md) | Automatic via `chat_workflow_cli/cli.py` scanning `workflows/` packages |
| Type safety | Document-level (DQS conformance) | Field-level (Python types, Pydantic validators) |

**Same purpose**: Structured artifact generation through AI-facilitated conversation.

**Different mechanism**: Deterministic Python validation replaces interpretive markdown. Typed return values replace document sections. The decorator replaces the playbook reader.

The prototype proved the idea. The chat-workflow library ships it.