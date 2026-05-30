# Coding Standards

These standards apply to all code in the chat-workflow library. They are designed to help readers (human or agent) understand what the code does and why it is structured that way.

## Types

- We use basedpyright to ensure comprehensive typing.
- Remember that the purpose of types is to help express to the reader (human or agent) what the code does. Make sure our typing is expressive. Multiple levels of nested dicts are not expressive.
- Always use the narrowest type that applies.
- If tempted to use "Any" or "object", double check whether a narrower type would be appropriate.
- If tempted to provide several types in a union, it's likely that a better approach would be to standardise on the one most appropriate type. If the current code uses a variety of types, don't automatically assume that this was a good idea.
- If tempted to put "| None" after your type, check that this isn't a cop-out. Are you sure we should really be allowing None?
- If we read in untyped data (for example, json as a string), coerce it to the narrow type as near to the edge as possible (i.e. in a cli or in unit tests). If we write untyped data, de-type it as close to the edge as possible.
- If tempted to #ignore a basedpyright error, think first. Is there a code or architecture smell that we should fix?

## Principles

- **Naming is very important.** Each class, function and variable should be named carefully in order to help readers understand the structure of the code as a whole.
- **Each class should be in its own module** named after that class.
- **Functionality that is tightly coupled to the contents of a class** should be a member function of that class.
- Code should be written in a **functional programming style** wherever reasonably possible.
- **Prefer libraries over reinvention**: Before writing non-trivial code from scratch, check whether a library already solves the problem. Adding a dev dependency has no user-facing cost. Adding a production dependency is often the right call too. The decision criterion is simplicity and readability: a library call that replaces 30 lines of custom code is worth it; a library that adds more complexity than the code it replaces is not.
- **Prefer to fail fast.** Don't silence errors, only use defaults where there is actually a good default option, don't have backstops, don't have three places that you look for something "just in case". Decide what should happen and then fail fast if it doesn't happen.
- **We do not maintain backwards compatibility** with previous versions of anything.
- **Import discipline**: Module and package exports should be organised so that the public API surface is importable from the package root. If code is moved to a different submodule, only ``__init__.py`` should need to change. External consumers must import from the package root (``from mypackage import Thing``), not from submodules (``from mypackage.submodule import Thing``). Internal code within the package should use relative submodule imports as normal.
- **Code generation safety**: Having code in strings within other code should be an absolute last resort when generating code.

### Class and Module Decomposition

Getting the class boundaries right is more important than picking the right name — a well-decomposed class is easy to name.

- **Classes represent things; functions represent actions.** A class name should be a noun or noun phrase drawn from the domain — something that exists in the problem space (``Session``, ``Workflow``, ``Criterion``, ``Config``). A function name should be a verb or verb phrase — something you do to or with those things (``process_turn``, ``normalized_weights``, ``validate_business_rules``). If you catch yourself naming a class with a verb-form like ``Orchestrator``, ``Manager``, ``Controller``, or ``Handler``, the class may not correspond to a real domain concept — you may be modelling an action as a thing. Consider whether the behaviour belongs as methods on a genuine domain noun instead.
- **Name things after what they are in the domain**, not after their structural role in the architecture. A class that IS a session should be named ``Session``, not ``SessionContext`` or ``SessionManager``. A class that IS a workflow should be named ``AtomicWorkflow``, not ``AtomicWorkflowOrchestrator``. Architectural suffixes ("Orchestrator", "Manager", "Context", "Handler", "Controller", "Tools") often mean you are describing how the code fits together rather than what the concept is. If the domain concept is clear, the name will be simple.
- **If a class or function name uses vague terms** like "Manager", "Enhanced" or "Configured", reconsider whether the base concept is well-defined. ``AtomicWorkflow`` without "Structured" says everything ``StructuredConversationOrchestrator`` said. When two concepts genuinely need disambiguation, the names should complement each other (e.g., ``AtomicWorkflow`` and ``CompositeWorkflow`` — each clarifies the other).
- **The docstring test**: If the best docstring you can write just rephrases the name (``"""ConversationOrchestrator orchestrates conversations."""``), that is a smell. Either the name is too vague or the concept boundaries are unclear.

### Module Naming

Avoid generic words like "utils", "manager", "tools" in module names. Use domain-driven names instead.

- ``prompt_builder.py`` not ``prompt_utils.py``
- ``metadata.py`` not ``utils/introspection.py``

A module named "utils" is a grab bag. It has no single responsibility. It grows without bound. Name modules after what they do.

### Prefer Flat Module Structure

Keep modules flat in the ``chat_workflow/`` directory rather than nesting them in subdirectories. Deep nesting hides information and makes imports harder to follow.

- ``metadata.py`` not ``utils/introspection.py``
- ``prompt_builder.py`` not ``prompt/prompt_builder.py``

### Single Responsibility Principle

Each module should have one reason to change.

- ``prompt_builder.py`` owns prompt formatting (docstring rendering, parameter section building)
- ``metadata.py`` owns type introspection (type name formatting, return type resolution, parameter inspection)
- ``atomic_workflow.py`` owns conversation orchestration (turn management, LLM calling, intent handling)
- ``llm_interaction.py`` owns LLM provider abstraction

If you find yourself adding a function to a module that doesn't match its stated purpose, create a new module.

### DRY: Extract Shared Logic

When the same pattern appears in multiple places, extract it into a dedicated module. The ``prompt_builder.py`` and ``metadata.py`` modules were extracted from ``atomic_workflow.py`` because prompt formatting and type introspection are used by ``decorators.py`` and are conceptually separate concerns.

## Smells

- **Circular import**: Fix the smell, don't bodge the import.
- **Long file**: A strong signal that multiple concerns have become mixed together. Identify subsets of the code that will change for different reasons and move each axis of change into its own module. Type-resolution logic changes when you add new type patterns; decorator logic changes when you alter the flow. Those belong in different files regardless of line count.
- **Circular docstring**: A docstring that adds no information beyond the name is a smell. The class may need a better name, clearer boundaries, or both. (Sometimes the class is really self describing with no need for a docstring, and that's great!)
