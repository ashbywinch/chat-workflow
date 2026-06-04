"""Unit test guard for LLM-judged validation rules.

``LLMValidated._skip_llm_validation`` defaults to ``False``, so
``validate_llm_rules`` WILL make an LLM call unless explicitly opted
out.  Unit tests MUST opt out by one of:

1. Use ``Model.model_construct(...)`` instead of ``Model(...)`` —
   bypasses all Pydantic validation (recommended for basic field tests).
2. Set ``_skip_llm_validation = True`` on the model class or subclass
   (use when you need ``@model_validator`` to run but not the LLM call).

Eval tests must NOT opt out — they exercise the real LLM validation path.
"""

