"""Called by .github/workflows/pr-agent.yml — fail the PR if the review bot
did not post a "PR Reviewer Guide" comment covering the head commit.
The review may have failed silently; this check prevents merging unreviewed.

A comment covers the head commit when its body references the commit's SHA
(the incremental-review form, "Starting from commit .../<SHA>") or it was
posted after the head commit landed (the first review on a PR is posted
without the SHA marker — observed: pr-agent v0.41.1 regular reviews never
contain the head SHA). The range-start SHA in the incremental body is the
first NEW commit, not the head — coverage still falls to the posted-after
rule; the explicit form is belt-and-braces for reviews that embed it.

The bot's own SKIP verdict also covers: an incremental review with no
files changed since the previous review posts "Incremental Review
Skipped" (linking the previous review) instead of a guide. That skip IS
the reviewed verdict — the diff since the last review was examined and
found empty (merging the base branch in, for example, changes nothing in
the PR's diff) — so it covers the commit exactly like the human /skip
opt-out does. A run that genuinely failed to post leaves no comment and
still fails.

Why the bot's own step cannot fail (2026-08-11, read from the v0.41.1
source): ``PRAgent.handle_request`` catches EVERY exception with a bare
``except`` — it logs "Failed to process the command." plus the traceback
and returns False — and ``github_action_runner.py`` discards that return
value, so the action exits 0 whether or not the review happened. A green
bot step is therefore meaningless, and the only truthful signal is this
check. To make the failure SELF-EXPLAINING rather than a guess, the check
fetches the run's own log (the checks API) and extracts the bot's error
lines: the "Failed to process the command." traceback, model/API errors,
and the diff-token-cap marker — the check's failure message names the
reason instead of "may have failed silently".
"""

import json
import os
import re
import sys
import urllib.request
from urllib.error import HTTPError

BOT_STEP_MARK = "Run the-pr-agent/pr-agent"

# The markers the bot leaves when a review dies inside the (swallowed)
# exception seam, plus the size signal that precedes a died model call.
_ERROR_MARKERS = (
    "Failed to process the command.",
    "Traceback (most recent call last)",
)
_SIZE_RE = re.compile(r"Tokens:\s*(\d+),\s*total tokens over limit:\s*(\d+)")
_TRACEBACK_TAIL = 12  # lines after a traceback/marker worth reporting


def _get(url: str, token: str, accept: str = "application/vnd.github+json") -> bytes:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": accept},
    )
    return urllib.request.urlopen(req).read()


def _get_json(url: str, token: str):
    return json.loads(_get(url, token))


# The log endpoint 302s to a signed blob URL; urllib's auto-follow forwards
# the Authorization header onto the blob host, which rejects it (401
# InvalidAuthenticationInfo, 2026-08-11). An opener with NO redirect handler
# stops at the 302 — HTTPErrorProcessor hands it to HTTPDefaultErrorHandler,
# which raises HTTPError, and the caller fetches the signed Location bare.
_OPENER = urllib.request.OpenerDirector()
for _handler in (urllib.request.HTTPHandler(), urllib.request.HTTPSHandler(),
                 urllib.request.HTTPErrorProcessor(), urllib.request.HTTPDefaultErrorHandler()):
    _OPENER.add_handler(_handler)


def _job_log(repo: str, token: str) -> str | None:
    """The running job's own accumulated log (the bot step's output), or
    None when the checks API won't serve it mid-run — the caller falls back
    to the generic message. Requires the workflow's ``actions: read``."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        return None
    try:
        jobs = _get_json(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs", token)
        if not jobs.get("jobs"):
            return None
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/actions/jobs/{jobs['jobs'][0]['id']}/logs",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        try:
            return _OPENER.open(req).read().decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.code != 302:
                raise
            location = e.headers.get("Location")
            if not location:
                return None
            return urllib.request.urlopen(location).read().decode("utf-8", errors="replace")
    except (HTTPError, KeyError, ValueError, urllib.error.URLError):
        return None


def _bot_failure_reason(repo: str, token: str) -> str:
    """The bot's own words about its failure, from the run log — the reason
    the review died, instead of a guess. Falls back to the last thing the
    bot said."""
    log = _job_log(repo, token)
    if log is None:
        # the fail-loud step runs inside the bot's OWN job — GitHub does not
        # serve a job's log until the job completes, so the reason extractor
        # can never see it here. Run this check from a SEPARATE job (needs:
        # the bot job) for the real reason; this fallback is the honest
        # admission that the review did not post.
        return "the bot's job log is not served while the job is still running (run this check from a separate job)"
    # The raw job blob is TIMESTAMPZ + message lines with no step prefixes
    # (the runner's processed view adds them); the bot's own records are the
    # structured JSON lines with a "text" field — those carry the markers.
    bot_lines = [ln for ln in log.splitlines() if '"text"' in ln]
    if not bot_lines:
        return "the bot's step produced no log output"

    return _failure_reason(bot_lines)


def _failure_reason(bot_lines: list[str]) -> str:
    """Why the bot died, from its own log lines: the diff-token cap (the
    size signal that precedes a died model call), the swallowed traceback
    seam, or the last thing it said."""
    for ln in bot_lines:
        m = _SIZE_RE.search(ln)
        if m:
            return (
                f"the PR's cumulative diff is {m.group(1)} tokens — over the bot's "
                f"{m.group(2)}-token review cap; the review died after pruning the diff"
            )
    for i, ln in enumerate(bot_lines):
        if any(mark in ln for mark in _ERROR_MARKERS):
            tail = [line for line in bot_lines[i : i + _TRACEBACK_TAIL] if line.strip()]
            snippet = " | ".join(line.split("\t")[-1][:200] for line in tail[:4])
            return f"the bot logged an error before dying: {snippet}"
    last = next((line for line in reversed(bot_lines) if line.strip()), "")
    return f"the bot's last output before going silent: {last.split(chr(9))[-1][:200]}"


def _review_covers(comments, sha: str, reviewed_after: str) -> bool:
    """Does any comment cover the head commit? A human /skip opt-out (applies
    to the whole PR), a posted guide (regular or incremental, referencing the
    sha or created after the head was received), or the bot's own
    incremental-skip verdict (created after the head was received — the diff
    was examined and found empty, so the skip is the review, not a
    failure)."""
    if any(c.get("body", "").strip() == "/skip" for c in comments):
        return True
    return any(
        (
            c.get("body", "").startswith(("## PR Reviewer Guide", "## Incremental PR Reviewer Guide"))
            and (
                sha in c.get("body", "")
                or f"commit/{sha}" in c.get("body", "")  # the incremental "Starting from commit .../<SHA>" form
                or c.get("created_at", "") >= reviewed_after
            )
        )
        or (
            c.get("body", "").startswith("Incremental Review Skipped")
            and (c.get("user") or {}).get("type") == "Bot"
            and c.get("created_at", "") >= reviewed_after
        )
        for c in comments
    )

def main() -> int:
    sha = os.environ["SHA"]
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]

    try:
        commit = _get_json(f"https://api.github.com/repos/{repo}/commits/{sha}", token)
    except HTTPError as e:
        # surfaces via the ::error line + the non-zero exit — not a swallow
        print(f"::error::commit {sha} is not fetchable ({e.code}) — the review cannot cover it.")
        return 1

    # "Reviewed after" is anchored to the workflow run's start, not the
    # commit's committer date: the committer date is author-controlled and
    # predates the push, so a review posted for an earlier revision could
    # "cover" a newer head authored before it. A review that covers THIS
    # push must post after the run that checks it started.
    reviewed_after = commit["commit"]["committer"]["date"]
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        try:
            run = _get_json(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}", token)
            reviewed_after = run["run_started_at"]
        # lucidlint: ignore swallow the run API is best-effort data availability — the ::warning print surfaces the degradation at runtime, and re-raising would fail the check when the committer-date anchor is still a conservative fallback
        except (HTTPError, KeyError, ValueError) as e:
            # fall back to the committer date — the run API may lag or the
            # token may lack actions: read. Never silent: a degraded anchor
            # is still a conservative one (committer date predates the push).
            print(f"::warning::run_started_at unavailable ({type(e).__name__}: {e}) — anchoring coverage to the committer date")

    try:
        comments = _get_json(f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100", token)
    except HTTPError as e:
        print(f"::error::PR comments are not fetchable ({e.code}) — the review coverage cannot be checked.")
        return 1
    if not _review_covers(comments, sha, reviewed_after):
        reason = _bot_failure_reason(repo, token)
        print(f"::error::AI review did not post for commit {sha} — {reason}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
