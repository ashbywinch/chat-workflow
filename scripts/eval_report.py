#!/usr/bin/env python3
"""Generate a human-readable time/cost report from eval-report.txt.

Usage:
    python scripts/eval_report.py [--report-path test-results/eval-report.txt]

Groups tests by category and shows subtotals plus estimated cost.
"""

import argparse
import re
from itertools import groupby
from pathlib import Path
from typing import NamedTuple


class EvalEntry(NamedTuple):
    test_name: str
    duration_s: float
    agent_tok: int
    judge_tok: int
    total_tok: int


class GroupedEntries(dict[str, list[EvalEntry]]):
    """Report group name -> the entries that fall in it."""


class TestGroups:
    """Name patterns mapped to groups for sensible subtotals."""

    PATTERNS: list[tuple[str, str]] = [
        (r"test_component_validation", "Component Validation"),
        (r"test_conversation_quality", "Conversation Quality"),
        (r"test_conversation_structure", "Conversation Structure"),
        (r"test_debug_streaming", "Debug Streaming"),
        (r"test_domain_exploration", "Domain Exploration"),
        (r"(test_end_to_end|^end_to_end)", "End-to-End Pipeline"),
        (r"test_generated_component_codegen", "Generated Component Codegen"),
        (r"test_generated_component_rules", "Generated Component Rules"),
        (r"test_interaction_gathering|gather_interaction", "Interaction Gathering"),
        (r"test_output_|test_resource_", "Output & Resource"),
        (r"test_process_definition", "Process Definition"),
        (r"test_real_api", "Real API"),
        (r"test_structural_design|design_from_domain", "Structural Design"),
        (r"test_workflow_evals", "Workflow Evals"),
        (r"test_workflow_pipeline", "Workflow Pipeline"),
        (r"multi_turn_component_design", "Generated Component Design"),
        (r"multi_turn_component_identification", "Component Identification"),
    ]

    @classmethod
    def for_test(cls, test_name: str) -> str:
        """Return the group whose pattern first matches *test_name*."""
        for pattern, group in cls.PATTERNS:
            if re.search(pattern, test_name):
                return group
        return "Other"


class TokenPricing:
    """Estimated cost per 1M tokens for the current model.

    Gemini 2.5 Flash Lite via OpenRouter: $0.10/1M input, $0.40/1M output,
    blended to ~$0.20/1M for a mixed input/output run.
    """

    BLENDED_PER_MILLION = 0.20


class ReportLayout:
    """Rule widths and unit scales for the plain-text report."""

    RULE_WIDTH = 80  # a full-width banner rule, in characters
    HEADER_COLUMN_RULES = (48, 7, 11, 11, 11, 10)  # dashes under each column
    GROUP_RULE = 70  # a group's subtotal rule, in characters
    SECONDS_PER_MINUTE = 60
    TOKENS_PER_MILLION = 1_000_000


def _banner(lines: list[str], title: str) -> None:
    """Append a full-width title banner to *lines*."""
    rule = "=" * ReportLayout.RULE_WIDTH
    lines.extend((rule, title, rule))


def _header_line() -> str:
    """Column headings for the per-test rows."""
    return (
        f"  {'Test':<50} {'Time(s)':>7} {'Agent(tok)':>11} {'Judge(tok)':>11} "
        f"{'Total(tok)':>11} {'Cost($)':>10}"
    )


def _dash_line() -> str:
    """Dashes under each heading in :func:`_header_line`."""
    rules = " ".join("-" * width for width in ReportLayout.HEADER_COLUMN_RULES)
    return f"  {rules}"


def _cost(tokens: int) -> float:
    """Blended dollar cost of *tokens*."""
    return (tokens / ReportLayout.TOKENS_PER_MILLION) * TokenPricing.BLENDED_PER_MILLION


def _entry_line(entry: EvalEntry) -> str:
    """One test's time, token and cost row."""
    return (
        f"  {entry.test_name:<50} {entry.duration_s:>6.0f}s {entry.agent_tok:>10,} "
        f"{entry.judge_tok:>10,} {entry.total_tok:>10,}  ${_cost(entry.total_tok):.4f}"
    )


def _group_section(group_name: str, group_entries: list[EvalEntry]) -> list[str]:
    """The heading, test rows and subtotal for one group."""
    total_s = sum(e.duration_s for e in group_entries)
    agent_tok = sum(e.agent_tok for e in group_entries)
    judge_tok = sum(e.judge_tok for e in group_entries)
    total_tok = sum(e.total_tok for e in group_entries)
    return [
        f"  ── {group_name} ──",
        *(_entry_line(e) for e in sorted(group_entries, key=lambda x: x.test_name)),
        f"  {'─' * ReportLayout.GROUP_RULE}",
        f"  {'Group total':<50} {total_s:>6.0f}s {agent_tok:>10,} "
        f"{judge_tok:>10,} {total_tok:>10,}  ${_cost(total_tok):.4f}",
        "",
    ]


def parse_report(report_path: Path) -> list[EvalEntry]:
    """Parse eval-report.txt.

    Lines are written by the @timeout decorator in conftest.py:
      [test_name] 10s  11243 tok
    """
    # Match simple:  [test_name] 10s  11243 tok
    pattern = re.compile(
        r"^\s+\[(?P<name>.+?)\]\s+(?P<seconds>[\d.]+)s\s+(?P<tokens>\d+)\s+tok"
    )

    # Dedup by taking the last occurrence per test name (in case of retries).
    entries: dict[str, EvalEntry] = {}
    for line in report_path.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name = m.group("name")
        tok = int(m.group("tokens"))
        entries[name] = EvalEntry(
            test_name=name,
            duration_s=float(m.group("seconds")),
            agent_tok=tok,
            judge_tok=0,
            total_tok=tok,
        )

    return list(entries.values())


def _grouped(entries: list[EvalEntry]) -> GroupedEntries:
    """Entries bucketed by report group name."""
    keyed = sorted(
        ((TestGroups.for_test(e.test_name), e) for e in entries),
        key=lambda pair: pair[0],
    )
    return GroupedEntries(
        {
            name: [entry for _, entry in bucket]
            for name, bucket in groupby(keyed, key=lambda pair: pair[0])
        }
    )


def generate_report(entries: list[EvalEntry], output_path: Path | None = None) -> str:
    """Build a human-readable report with groups and subtotals."""
    groups = _grouped(entries)

    lines: list[str] = []
    _banner(lines, "EVAL TIME & COST REPORT")
    lines.extend(("", _header_line(), _dash_line(), ""))
    lines.extend(
        line
        for name in sorted(groups)
        for line in _group_section(name, groups[name])
    )

    grand_total_s = sum(e.duration_s for e in entries)
    grand_agent_tok = sum(e.agent_tok for e in entries)
    grand_judge_tok = sum(e.judge_tok for e in entries)
    grand_total_tok = sum(e.total_tok for e in entries)

    _banner(lines, "GRAND TOTALS")
    lines.append(
        f"  Total time:       {grand_total_s:.0f}s "
        f"({grand_total_s / ReportLayout.SECONDS_PER_MINUTE:.1f} min)"
    )
    lines.append(f"  Agent tokens:     {grand_agent_tok:>10,}")
    lines.append(f"  Judge tokens:     {grand_judge_tok:>10,}")
    lines.append(f"  Total tokens:     {grand_total_tok:>10,}")
    lines.append(f"  Estimated cost:   ${_cost(grand_total_tok):.4f}")
    lines.append(
        f"  (at ${TokenPricing.BLENDED_PER_MILLION:.2f}/1M tokens blended rate)"
    )
    lines.append("")
    lines.append(f"  Tests included:   {len(entries)}")
    lines.append(f"  Groups:           {len(groups)}")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval time/cost report")
    parser.add_argument("--report-path", default="test-results/eval-report.txt",
                        help="Path to eval-report.txt")
    parser.add_argument("--output", default="test-results/cost-report.txt",
                        help="Path to write the report")
    args = parser.parse_args()

    report_path = Path(args.report_path)
    if not report_path.exists():
        print(f"Report file not found: {report_path}")
        return

    entries = parse_report(report_path)
    if not entries:
        print(f"No eval entries found in {report_path}")
        return

    generate_report(entries, Path(args.output))


if __name__ == "__main__":
    main()
