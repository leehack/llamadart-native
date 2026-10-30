"""Read job and step structure from workflow text; the runners have no PyYAML."""

from __future__ import annotations

from pathlib import Path
import re

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"


def job_block(workflow: str, job: str) -> str:
    text = (WORKFLOWS / workflow).read_text()
    block = re.search(rf"^  {re.escape(job)}:\n((?:(?:    .*)?\n)+)", text, re.MULTILINE)
    if not block:
        raise AssertionError(f"{workflow} has no job {job}")
    return block[1]


def job_keys(block: str) -> list[str]:
    return re.findall(r"^    ([a-z-]+):", block, re.MULTILINE)


def steps(block: str) -> list[str]:
    body = block.split("\n    steps:\n", 1)[1]
    return [f"      - {step.rstrip()}\n" for step in re.split(r"^      - ", body, flags=re.MULTILINE)[1:]]


def step_keys(step: str) -> list[str]:
    return re.findall(r"^(?:      - |        )([a-z-]+):", step, re.MULTILINE)


def step_name(step: str) -> str:
    name = re.search(r"^(?:      - |        )name: (.+)$", step, re.MULTILINE)
    uses = re.search(r"^(?:      - |        )uses: (.+)$", step, re.MULTILINE)
    return name[1] if name else uses[1]


def step_running(block: str, command: str) -> str:
    matches = [step for step in steps(block) if command in step]
    if len(matches) != 1:
        raise AssertionError(f"expected one step running {command}, found {len(matches)}")
    return matches[0]


def matrix_rows(block: str) -> list[dict[str, str]]:
    include = block.split("        include:\n", 1)[1].split("\n    steps:\n", 1)[0]
    rows: list[dict[str, str]] = []
    for line in include.splitlines():
        entry = re.fullmatch(r"          (- |  )([a-z_]+): (.+)", line)
        if not entry:
            continue
        if entry[1] == "- ":
            rows.append({})
        rows[-1][entry[2]] = entry[3]
    return rows
