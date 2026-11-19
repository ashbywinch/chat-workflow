#!/usr/bin/env python3
"""Generate a human-readable time/cost report from eval-report.txt.

Usage:
    python scripts/eval_report.py [--report-path test-results/eval-report.txt]

Groups tests by category and shows subtotals plus estimated cost.
"""

import argparse
import os
import re
from pathlib import Path
from typing import NamedTuple


class EvalEntry(NamedTuple):
    test_name: str
    duration_s: float
    agent_tok: int
    judge_tok: int
    total_tok: int


# Map test names to groups for sensible subtotals.
GROUP_PATTERNS: list[tuple[str, str]] = [
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


def _group_name(test_name: str) -> str:
    for pattern, group in GROUP_PATTERNS:
        if re.search(pattern, test_name):
            return group
    return "Other"


# Estimated cost per 1M tokens for the current model
# Gemini 2.5 Flash Lite via OpenRouter: $0.10/1M input, $0.40/1M output
# Using a blended rate of ~$0.20/1M for mixed input/output
COST_PER_1M_TOKENS = 0.20  # blended input + output


def parse_report(report_path: Path) -> list[EvalEntry]:
    """Parse eval-report.txt.

    Lines are written by the @timeout decorator in conftest.py:
      [test_name] 10s  11243 tok
    """
    # Match simple:  [test_name] 10s  11243 tok
    pattern = re.compile(
        r"^\s+\[(.+?)\]\s+([\d.]+)s\s+(\d+)\s+tok"
    )

    # Dedup by taking the last occurrence per test name (in case of retries).
    entries: dict[str, EvalEntry] = {}
    for line in report_path.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name = m.group(1)
        tok = int(m.group(3))
        entries[name] = EvalEntry(
            test_name=name,
            duration_s=float(m.group(2)),
            agent_tok=tok,
            judge_tok=0,
            total_tok=tok,
        )

    return list(entries.values())


def generate_report(entries: list[EvalEntry], output_path: Path | None = None) -> str:
    """Build a human-readable report with groups and subtotals."""
    # Group entries
    groups: dict[str, list[EvalEntry]] = {}
    for e in entries:
        g = _group_name(e.test_name)
        groups.setdefault(g, []).append(e)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("EVAL TIME & COST REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Test':<50} {'Time(s)':>7} {'Agent(tok)':>11} {'Judge(tok)':>11} {'Total(tok)':>11} {'Cost($)':>10}")
    lines.append(f"  {'-'*48} {'-'*7} {'-'*11} {'-'*11} {'-'*11} {'-'*10}")
    lines.append("")

    grand_total_s = 0.0
    grand_agent_tok = 0
    grand_judge_tok = 0
    grand_total_tok = 0

    for group_name in sorted(groups.keys()):
        group_entries = groups[group_name]
        group_total_s = sum(e.duration_s for e in group_entries)
        group_agent_tok = sum(e.agent_tok for e in group_entries)
        group_judge_tok = sum(e.judge_tok for e in group_entries)
        group_total_tok = sum(e.total_tok for e in group_entries)
        group_cost = (group_total_tok / 1_000_000) * COST_PER_1M_TOKENS

        grand_total_s += group_total_s
        grand_agent_tok += group_agent_tok
        grand_judge_tok += group_judge_tok
        grand_total_tok += group_total_tok

        lines.append(f"  ── {group_name} ──")

        for e in sorted(group_entries, key=lambda x: x.test_name):
            ecost = (e.total_tok / 1_000_000) * COST_PER_1M_TOKENS
            lines.append(
                f"  {e.test_name:<50} {e.duration_s:>6.0f}s {e.agent_tok:>10,} {e.judge_tok:>10,} {e.total_tok:>10,}  ${ecost:.4f}"
            )

        lines.append(f"  {'─'*70}")
        lines.append(f"  {'Group total':<50} {group_total_s:>6.0f}s {group_agent_tok:>10,} {group_judge_tok:>10,} {group_total_tok:>10,}  ${group_cost:.4f}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("GRAND TOTALS")
    lines.append("=" * 80)
    grand_cost = (grand_total_tok / 1_000_000) * COST_PER_1M_TOKENS
    lines.append(f"  Total time:       {grand_total_s:.0f}s ({grand_total_s/60:.1f} min)")
    lines.append(f"  Agent tokens:     {grand_agent_tok:>10,}")
    lines.append(f"  Judge tokens:     {grand_judge_tok:>10,}")
    lines.append(f"  Total tokens:     {grand_total_tok:>10,}")
    lines.append(f"  Estimated cost:   ${grand_cost:.4f}")
    lines.append(f"  (at ${COST_PER_1M_TOKENS:.2f}/1M tokens blended rate)")
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
