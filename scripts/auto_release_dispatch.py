#!/usr/bin/env python3
"""Plan automatic preparation and owner approval for stable native candidates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from release_contract import (
    DISCOVERY_STATUSES,
    ContractError,
    build_discovery_report,
    validate_dispatch,
)


CORRELATION_NAMESPACE = "auto-stable"
MAX_CORRELATION_LENGTH = 128
DISPATCH_WORKFLOW = "native_release.yml"
SMOKE_POLICY = "required"
PREPARATION_PUBLISH_RELEASE = False
APPROVAL_PUBLISH_RELEASE = True
# An `incompatible` upstream ref is reported verbatim without passing the tag
# grammar, so its shape is constrained here before it reaches step outputs.
UPSTREAM_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def deterministic_correlation_id(
    *, upstream_ref: str, upstream_commit: str, native_release_tag: str
) -> str:
    """Return a correlation identifier bound only to the exact release identity.

    Retries and later scheduled runs must reuse the same identifier so
    `native_release.yml` reconciles them as one idempotent publication
    transaction; nothing run-specific may enter the digest.
    """
    digest = hashlib.sha256(
        "\n".join(
            (
                CORRELATION_NAMESPACE,
                upstream_ref,
                upstream_commit.lower(),
                native_release_tag,
            )
        ).encode("utf-8")
    ).hexdigest()
    prefix = f"{CORRELATION_NAMESPACE}/{native_release_tag}/"
    return prefix + digest[: MAX_CORRELATION_LENGTH - len(prefix)]


def _plan(
    *,
    status: str,
    decision: str,
    in_flight_native_runs: int,
    reason: str,
    dispatch: dict[str, Any] | None,
    approval: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "decision": decision,
        "in_flight_native_runs": in_flight_native_runs,
        "reason": reason,
        "dispatch": dispatch,
        "approval": approval,
    }


def workflow_dispatch_inputs(dispatch_plan: Mapping[str, Any]) -> dict[str, str]:
    """Render a typed plan as the string map required by ``gh workflow run --json``."""
    dispatch = dispatch_plan.get("dispatch")
    if not isinstance(dispatch, Mapping):
        raise ContractError("dispatch inputs require an authorized dispatch plan")
    inputs = dispatch.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("authorized dispatch plan is missing its input object")

    rendered: dict[str, str] = {}
    for key, value in inputs.items():
        if not isinstance(key, str) or not key:
            raise ContractError("workflow dispatch input names must be non-empty strings")
        if isinstance(value, bool):
            rendered[key] = "true" if value else "false"
        elif isinstance(value, str):
            rendered[key] = value
        else:
            raise ContractError(
                f"workflow dispatch input {key!r} must be a string or boolean"
            )
    return rendered


def _publication_approval(contract: Mapping[str, str]) -> dict[str, Any]:
    return {
        "workflow": DISPATCH_WORKFLOW,
        "required_actor": "repository-owner",
        "inputs": {
            "llama_cpp_tag": contract["llama_cpp_ref"],
            "llama_cpp_commit": contract["llama_cpp_commit"],
            "native_source_sha": contract["native_source_sha"],
            "native_release_tag": contract["native_release_tag"],
            "smoke_policy": SMOKE_POLICY,
            "correlation_id": contract["correlation_id"],
            "publish_release": APPROVAL_PUBLISH_RELEASE,
        },
    }


def build_dispatch_plan(
    *,
    status: str,
    upstream_ref: str,
    upstream_commit: str,
    in_flight_native_runs: int,
    native_head_commit: str,
    policy_error: str = "",
) -> dict[str, Any]:
    """Decide whether discovery state authorizes exactly one native release dispatch."""
    if status not in DISCOVERY_STATUSES:
        raise ContractError(f"discovery status must be one of {DISCOVERY_STATUSES}")
    if UPSTREAM_REF_RE.fullmatch(upstream_ref) is None:
        raise ContractError(
            "upstream release ref must be 1-64 characters, start with an alphanumeric "
            "character, and contain only alphanumerics, '.', '_', or '-'"
        )
    if isinstance(in_flight_native_runs, bool) or not isinstance(in_flight_native_runs, int):
        raise ContractError("in-flight native release run count must be an integer")
    if in_flight_native_runs < 0:
        raise ContractError("in-flight native release run count must not be negative")

    native_release_tag = upstream_ref

    if status == "incompatible":
        detail = policy_error.strip() or "upstream release ref is outside the stable channel"
        return _plan(
            status=status,
            decision="fail",
            in_flight_native_runs=in_flight_native_runs,
            reason=(
                "Latest upstream release is incompatible with the stable native "
                f"release policy: {detail}"
            ),
            dispatch=None,
            approval=None,
        )

    correlation_id = deterministic_correlation_id(
        upstream_ref=upstream_ref,
        upstream_commit=upstream_commit,
        native_release_tag=native_release_tag,
    )
    preparation_contract = validate_dispatch(
        upstream_ref=upstream_ref,
        upstream_commit=upstream_commit,
        native_source_sha=native_head_commit,
        native_release_tag=native_release_tag,
        smoke_policy=SMOKE_POLICY,
        correlation_id=correlation_id,
        publish_release=PREPARATION_PUBLISH_RELEASE,
    )
    approval_contract = validate_dispatch(
        upstream_ref=upstream_ref,
        upstream_commit=upstream_commit,
        native_source_sha=native_head_commit,
        native_release_tag=native_release_tag,
        smoke_policy=SMOKE_POLICY,
        correlation_id=correlation_id,
        publish_release=APPROVAL_PUBLISH_RELEASE,
    )
    if (
        preparation_contract["upstream_channel"] != "stable"
        or preparation_contract["release_kind"] != "upstream"
    ):
        raise ContractError(
            "automatic dispatch is limited to exact upstream-aligned stable releases"
        )

    if status == "noop":
        return _plan(
            status=status,
            decision="skip",
            in_flight_native_runs=in_flight_native_runs,
            reason=(
                f"Native release {native_release_tag} already exists; no dispatch, "
                "publication, or submodule mutation was attempted."
            ),
            dispatch=None,
            approval=None,
        )

    if in_flight_native_runs > 0:
        return _plan(
            status=status,
            decision="skip",
            in_flight_native_runs=in_flight_native_runs,
            reason=(
                f"Native release {native_release_tag} is unbuilt, but "
                f"{in_flight_native_runs} native release run(s) are already queued or "
                "in progress; no duplicate dispatch was issued."
            ),
            dispatch=None,
            approval=_publication_approval(approval_contract),
        )

    return _plan(
        status=status,
        decision="prepare",
        in_flight_native_runs=in_flight_native_runs,
        reason=(
            f"Unbuilt stable upstream release {upstream_ref} detected; the exact native "
            f"non-publishing preparation for {native_release_tag} is authorized under "
            f"correlation {preparation_contract['correlation_id']}; publication still "
            "requires one explicit repository-owner workflow dispatch."
        ),
        dispatch={
            "workflow": DISPATCH_WORKFLOW,
            "inputs": {
                "llama_cpp_tag": preparation_contract["llama_cpp_ref"],
                "llama_cpp_commit": preparation_contract["llama_cpp_commit"],
                "native_source_sha": preparation_contract["native_source_sha"],
                "native_release_tag": preparation_contract["native_release_tag"],
                "smoke_policy": SMOKE_POLICY,
                "correlation_id": preparation_contract["correlation_id"],
                "publish_release": PREPARATION_PUBLISH_RELEASE,
            },
        },
        approval=_publication_approval(approval_contract),
    )


def _write_json(payload: Mapping[str, Any], output: Path) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise ContractError(f"unable to write {output}: {error}") from error


def _remove_stale_outputs(outputs: tuple[Path, ...]) -> None:
    try:
        for output in outputs:
            output.unlink(missing_ok=True)
    except OSError as error:
        raise ContractError(f"unable to remove stale output {output}: {error}") from error


def _non_negative_int(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError(
            "in-flight native release run count must be a non-negative integer"
        )
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--status", required=True, choices=DISCOVERY_STATUSES)
    plan.add_argument("--upstream-ref", required=True)
    plan.add_argument("--upstream-commit", required=True)
    plan.add_argument("--native-head-commit", required=True)
    plan.add_argument("--workflow-run-id", required=True)
    plan.add_argument("--workflow-run-url", required=True)
    plan.add_argument("--policy-error", default="")
    plan.add_argument("--in-flight-native-runs", required=True, type=_non_negative_int)
    plan.add_argument("--report-output", required=True, type=Path)
    plan.add_argument("--plan-output", required=True, type=Path)
    plan.add_argument("--dispatch-inputs-output", required=True, type=Path)

    args = parser.parse_args()
    try:
        _remove_stale_outputs(
            (
                args.report_output,
                args.plan_output,
                args.dispatch_inputs_output,
            )
        )
        dispatch_plan = build_dispatch_plan(
            status=args.status,
            upstream_ref=args.upstream_ref,
            upstream_commit=args.upstream_commit,
            native_head_commit=args.native_head_commit,
            in_flight_native_runs=args.in_flight_native_runs,
            policy_error=args.policy_error,
        )
        report = build_discovery_report(
            status=args.status,
            upstream_ref=args.upstream_ref,
            upstream_commit=args.upstream_commit,
            native_head_commit=args.native_head_commit,
            workflow_run_id=args.workflow_run_id,
            workflow_run_url=args.workflow_run_url,
            message=dispatch_plan["reason"],
        )
        report["decision"] = dispatch_plan["decision"]
        report["in_flight_native_runs"] = dispatch_plan["in_flight_native_runs"]
        report["dispatch"] = dispatch_plan["dispatch"]
        report["approval"] = dispatch_plan["approval"]

        _write_json(report, args.report_output)
        _write_json(dispatch_plan, args.plan_output)
        if dispatch_plan["dispatch"] is not None:
            _write_json(
                workflow_dispatch_inputs(dispatch_plan), args.dispatch_inputs_output
            )
    except ContractError as error:
        parser.error(str(error))

    outputs = {
        "status": args.status,
        "decision": dispatch_plan["decision"],
        "native_release_tag": args.upstream_ref,
        "tag": args.upstream_ref,
        "llama_cpp_ref": args.upstream_ref,
        "llama_cpp_commit": report["llama_cpp_commit"],
        "in_flight_native_runs": str(dispatch_plan["in_flight_native_runs"]),
        "correlation_id": (
            dispatch_plan["dispatch"]["inputs"]["correlation_id"]
            if dispatch_plan["dispatch"] is not None
            else dispatch_plan["approval"]["inputs"]["correlation_id"]
            if dispatch_plan["approval"] is not None
            else ""
        ),
    }
    # Step outputs are emitted only after every required report and dispatch
    # input file has been written, so the workflow cannot act on partial state.
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
