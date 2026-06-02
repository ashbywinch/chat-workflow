"""Unit test guard for LLM-judged validation rules.

``LLMValidated._skip_llm_validation`` defaults to ``True``, so
``validate_llm_rules`` silently returns ``self`` whenever ``unittest``
is in ``sys.modules`` — no real LLM API calls during unit tests.

Tests that need to exercise ``validate_llm_rules`` with a mocked LLM
should set ``_skip_llm_validation = False`` on their model subclass.
"""

