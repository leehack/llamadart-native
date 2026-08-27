from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from auto_release_dispatch import (  # noqa: E402
    build_dispatch_plan,
    deterministic_correlation_id,
    workflow_dispatch_inputs,
)
from release_contract import CORRELATION_RE, ContractError  # noqa: E402
from verify_release_provenance import workflow_run_blocks  # noqa: E402


AUTO_WORKFLOW = ROOT / ".github/workflows/auto_native_release.yml"
UPSTREAM_COMMIT = "a" * 40
NATIVE_HEAD_COMMIT = "b" * 40

FAKE_GH = '''#!/usr/bin/env python3
import json
import os
import sys

argv = sys.argv[1:]
payload = sys.stdin.read() if "--json" in argv else None
previous_calls = []
if os.path.exists(os.environ["FAKE_GH_LOG"]):
    with open(os.environ["FAKE_GH_LOG"], encoding="utf-8") as handle:
        previous_calls = [json.loads(line) for line in handle if line.strip()]
with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": argv, "stdin": payload}) + "\\n")

if argv[:1] == ["api"]:
    target = next((value for value in argv[1:] if value.startswith("repos/")), "")
    if target == "repos/ggml-org/llama.cpp/releases/latest":
        if os.environ.get("FAKE_LATEST_RELEASE_ERROR"):
            print(os.environ["FAKE_LATEST_RELEASE_ERROR"], file=sys.stderr)
            raise SystemExit(1)
        print(os.environ["FAKE_UPSTREAM_REF"])
        raise SystemExit(0)
    if target.startswith("repos/ggml-org/llama.cpp/commits/"):
        prior = sum(
            call["argv"][:1] == ["api"]
            and any(value.startswith("repos/ggml-org/llama.cpp/commits/") for value in call["argv"])
            for call in previous_calls
        )
        error = (
            os.environ.get("FAKE_COMMIT_LOOKUP_ERROR_AFTER_PLAN", "")
            if prior
            else os.environ.get("FAKE_COMMIT_LOOKUP_ERROR", "")
        )
        if error:
            print(error, file=sys.stderr)
            raise SystemExit(1)
        if prior and os.environ.get("FAKE_UPSTREAM_COMMIT_AFTER_PLAN"):
            print(os.environ["FAKE_UPSTREAM_COMMIT_AFTER_PLAN"])
        else:
            print(os.environ["FAKE_UPSTREAM_COMMIT"])
        raise SystemExit(0)
    if "/releases/tags/" in target:
        prior = sum(
            call["argv"][:1] == ["api"]
            and any("/releases/tags/" in value for value in call["argv"])
            for call in previous_calls
        )
        error = (
            os.environ.get("FAKE_RELEASE_LOOKUP_ERROR_AFTER_PLAN", "")
            if prior
            else os.environ.get("FAKE_RELEASE_LOOKUP_ERROR", "")
        )
        if error:
            print(error, file=sys.stderr)
            raise SystemExit(1)
        exists = os.environ["FAKE_NATIVE_RELEASE_EXISTS"] == "true"
        if prior and os.environ.get("FAKE_NATIVE_RELEASE_EXISTS_AFTER_PLAN"):
            exists = os.environ["FAKE_NATIVE_RELEASE_EXISTS_AFTER_PLAN"] == "true"
        if exists:
            raise SystemExit(0)
        print("gh: Not Found (HTTP 404)", file=sys.stderr)
        raise SystemExit(1)
    if "/actions/workflows/native_release.yml/runs" in target:
        if "--paginate" not in argv or "--slurp" not in argv:
            print("in-flight query must inspect every result page", file=sys.stderr)
            raise SystemExit(1)
        if "--jq" in argv or "--template" in argv:
            print("--slurp must feed an external jq process", file=sys.stderr)
            raise SystemExit(1)
        prior = sum(
            call["argv"][:1] == ["api"]
            and any("/actions/workflows/native_release.yml/runs" in value for value in call["argv"])
            for call in previous_calls
        )
        error = (
            os.environ.get("FAKE_RUNS_LOOKUP_ERROR_AFTER_PLAN", "")
            if prior
            else os.environ.get("FAKE_RUNS_LOOKUP_ERROR", "")
        )
        if error:
            print(error, file=sys.stderr)
            raise SystemExit(1)
        if prior and os.environ.get("FAKE_NATIVE_RUN_STATUSES_AFTER_PLAN"):
            statuses = json.loads(os.environ["FAKE_NATIVE_RUN_STATUSES_AFTER_PLAN"])
        else:
            statuses = json.loads(os.environ["FAKE_NATIVE_RUN_STATUSES"])
        print(
            json.dumps(
                [{"workflow_runs": [{"status": status} for status in statuses]}]
            )
        )
        raise SystemExit(0)
    print(f"unexpected gh api target: {target}", file=sys.stderr)
    raise SystemExit(1)

if argv[:2] == ["workflow", "run"]:
    try:
        inputs = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        print(f"could not parse provided JSON: {error}", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(inputs, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in inputs.items()
    ):
        print("could not parse provided JSON as map[string]string", file=sys.stderr)
        raise SystemExit(1)
    if os.environ.get("FAKE_DISPATCH_ERROR"):
        print(os.environ["FAKE_DISPATCH_ERROR"], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

print(f"unexpected gh invocation: {argv}", file=sys.stderr)
raise SystemExit(1)
'''


class OrchestrationRun:
    def __init__(self, process: subprocess.CompletedProcess[str], root: Path, log: Path) -> None:
        self.process = process
        self.root = root
        self.gh_calls = [
            json.loads(line) for line in log.read_text().splitlines() if line.strip()
        ]
        self.dispatches = [
            call for call in self.gh_calls if call["argv"][:2] == ["workflow", "run"]
        ]
        report = root / "native-discovery-report.json"
        plan = root / "native-dispatch-plan.json"
        self.report = json.loads(report.read_text()) if report.is_file() else None
        self.plan = json.loads(plan.read_text()) if plan.is_file() else None
        self.outputs = dict(
            line.split("=", 1)
            for line in (root / "github_output").read_text().splitlines()
            if "=" in line
        )

    @property
    def dispatch_inputs(self) -> dict[str, object]:
        self_dispatch = self.dispatches[0]
        return json.loads(self_dispatch["stdin"])


class AutoReleaseDispatchPlanTests(unittest.TestCase):
    def test_correlation_id_is_deterministic_and_contract_safe(self) -> None:
        correlation = deterministic_correlation_id(
            upstream_ref="v9.9.9",
            upstream_commit=UPSTREAM_COMMIT,
            native_release_tag="v9.9.9",
        )
        digest = hashlib.sha256(
            f"auto-stable\nv9.9.9\n{UPSTREAM_COMMIT}\nv9.9.9".encode()
        ).hexdigest()
        self.assertEqual(f"auto-stable/v9.9.9/{digest}", correlation)
        self.assertIsNotNone(CORRELATION_RE.fullmatch(correlation))
        self.assertEqual(
            correlation,
            deterministic_correlation_id(
                upstream_ref="v9.9.9",
                upstream_commit=UPSTREAM_COMMIT.upper(),
                native_release_tag="v9.9.9",
            ),
        )
        self.assertNotEqual(
            correlation,
            deterministic_correlation_id(
                upstream_ref="v9.9.10",
                upstream_commit=UPSTREAM_COMMIT,
                native_release_tag="v9.9.10",
            ),
        )
        longest_tag = "v" + "9" * 20 + "." + "9" * 20 + "." + "9" * 20
        longest = deterministic_correlation_id(
            upstream_ref=longest_tag,
            upstream_commit=UPSTREAM_COMMIT,
            native_release_tag=longest_tag,
        )
        self.assertEqual(128, len(longest))
        self.assertIsNotNone(CORRELATION_RE.fullmatch(longest))

    def test_candidate_plan_carries_the_exact_publication_contract(self) -> None:
        plan = build_dispatch_plan(
            status="candidate",
            upstream_ref="v9.9.9",
            upstream_commit=UPSTREAM_COMMIT.upper(),
            in_flight_native_runs=0,
        )
        self.assertEqual("dispatch", plan["decision"])
        self.assertEqual("native_release.yml", plan["dispatch"]["workflow"])
        self.assertEqual(
            {
                "llama_cpp_tag": "v9.9.9",
                "llama_cpp_commit": UPSTREAM_COMMIT,
                "native_release_tag": "v9.9.9",
                "smoke_policy": "required",
                "correlation_id": deterministic_correlation_id(
                    upstream_ref="v9.9.9",
                    upstream_commit=UPSTREAM_COMMIT,
                    native_release_tag="v9.9.9",
                ),
                "publish_release": True,
            },
            plan["dispatch"]["inputs"],
        )
        self.assertIs(True, plan["dispatch"]["inputs"]["publish_release"])
        self.assertEqual(
            "true", workflow_dispatch_inputs(plan)["publish_release"]
        )
        self.assertTrue(
            all(isinstance(value, str) for value in workflow_dispatch_inputs(plan).values())
        )

    def test_noop_and_in_flight_plans_never_dispatch(self) -> None:
        for status, in_flight in (("noop", 0), ("candidate", 1), ("candidate", 7)):
            with self.subTest(status=status, in_flight=in_flight):
                plan = build_dispatch_plan(
                    status=status,
                    upstream_ref="v9.9.9",
                    upstream_commit=UPSTREAM_COMMIT,
                    in_flight_native_runs=in_flight,
                )
                self.assertEqual("skip", plan["decision"])
                self.assertIsNone(plan["dispatch"])

    def test_incompatible_plan_fails_closed_with_its_diagnostic(self) -> None:
        plan = build_dispatch_plan(
            status="incompatible",
            upstream_ref="b10545",
            upstream_commit=UPSTREAM_COMMIT,
            in_flight_native_runs=0,
            policy_error="automatic release discovery requires stable vMAJOR.MINOR.PATCH",
        )
        self.assertEqual("fail", plan["decision"])
        self.assertIsNone(plan["dispatch"])
        self.assertIn("requires stable vMAJOR.MINOR.PATCH", plan["reason"])

    def test_non_stable_or_wrapper_candidates_are_rejected(self) -> None:
        for upstream_ref in ("b10545", "v9.9.9-1", "latest"):
            with self.subTest(upstream_ref=upstream_ref):
                with self.assertRaises(ContractError):
                    build_dispatch_plan(
                        status="candidate",
                        upstream_ref=upstream_ref,
                        upstream_commit=UPSTREAM_COMMIT,
                        in_flight_native_runs=0,
                    )

    def test_invalid_discovery_state_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            build_dispatch_plan(
                status="unknown",
                upstream_ref="v9.9.9",
                upstream_commit=UPSTREAM_COMMIT,
                in_flight_native_runs=0,
            )
        with self.assertRaises(ContractError):
            build_dispatch_plan(
                status="candidate",
                upstream_ref="v9.9.9",
                upstream_commit=UPSTREAM_COMMIT,
                in_flight_native_runs=-1,
            )
        with self.assertRaises(ContractError):
            build_dispatch_plan(
                status="candidate",
                upstream_ref="v9.9.9",
                upstream_commit="abc",
                in_flight_native_runs=1,
            )
        with self.assertRaises(ContractError):
            build_dispatch_plan(
                status="noop",
                upstream_ref="b10545",
                upstream_commit=UPSTREAM_COMMIT,
                in_flight_native_runs=0,
            )

    def test_unsafe_upstream_ref_never_reaches_step_outputs(self) -> None:
        for upstream_ref in (
            "v9.9.9\nstatus=candidate",
            "v9.9.9 && touch pwned",
            "'; touch pwned; '",
            "",
            "v" * 65,
        ):
            for status in ("candidate", "noop", "incompatible"):
                with self.subTest(upstream_ref=upstream_ref, status=status):
                    with self.assertRaises(ContractError):
                        build_dispatch_plan(
                            status=status,
                            upstream_ref=upstream_ref,
                            upstream_commit=UPSTREAM_COMMIT,
                            in_flight_native_runs=0,
                            policy_error="unrecognized upstream ref",
                        )

    def test_failed_planning_removes_every_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [
                root / "report.json",
                root / "plan.json",
                root / "inputs.json",
            ]
            for output in outputs:
                output.write_text("stale\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/auto_release_dispatch.py"),
                    "plan",
                    "--status",
                    "candidate",
                    "--upstream-ref",
                    "v9.9.9",
                    "--upstream-commit",
                    "not-a-commit",
                    "--native-head-commit",
                    NATIVE_HEAD_COMMIT,
                    "--workflow-run-id",
                    "42",
                    "--workflow-run-url",
                    "https://github.com/example/actions/runs/42",
                    "--in-flight-native-runs",
                    "0",
                    "--report-output",
                    str(outputs[0]),
                    "--plan-output",
                    str(outputs[1]),
                    "--dispatch-inputs-output",
                    str(outputs[2]),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertTrue(all(not output.exists() for output in outputs))


class AutoReleaseDispatchWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workflow = AUTO_WORKFLOW.read_text()
        discovery_blocks = [
            block
            for block in workflow_run_blocks(workflow)
            if "scripts/auto_release_dispatch.py plan" in block
        ]
        dispatch_blocks = [
            block
            for block in workflow_run_blocks(workflow)
            if "gh workflow run native_release.yml" in block
        ]
        failure_blocks = [
            block
            for block in workflow_run_blocks(workflow)
            if "Latest upstream release is incompatible; see" in block
        ]
        if not all(
            len(blocks) == 1
            for blocks in (discovery_blocks, dispatch_blocks, failure_blocks)
        ):
            raise AssertionError(
                "auto_native_release.yml must contain one discovery, dispatch, and "
                "fail-closed run block"
            )
        cls.discovery_block = discovery_blocks[0]
        cls.dispatch_block = dispatch_blocks[0]
        cls.failure_block = failure_blocks[0]

    def run_orchestration(
        self,
        *,
        upstream_ref: str = "v9.9.9",
        upstream_commit: str = UPSTREAM_COMMIT,
        native_release_exists: bool = False,
        native_release_exists_after_plan: bool | None = None,
        native_run_statuses: tuple[str, ...] = (),
        native_run_statuses_after_plan: tuple[str, ...] | None = None,
        upstream_commit_after_plan: str = "",
        latest_release_error: str = "",
        commit_lookup_error: str = "",
        commit_lookup_error_after_plan: str = "",
        release_lookup_error: str = "",
        release_lookup_error_after_plan: str = "",
        runs_lookup_error: str = "",
        runs_lookup_error_after_plan: str = "",
        dispatch_error: str = "",
        seed_stale_outputs: bool = False,
        mutate_dispatch_input: tuple[str, str] | None = None,
    ) -> OrchestrationRun:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        shutil.copytree(
            ROOT / "scripts",
            root / "scripts",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake_gh = bin_dir / "gh"
        fake_gh.write_text(FAKE_GH, encoding="utf-8")
        fake_gh.chmod(0o755)
        log = root / "gh.log"
        log.write_text("", encoding="utf-8")
        output = root / "github_output"
        output.write_text("", encoding="utf-8")
        if seed_stale_outputs:
            for name in (
                "native-discovery-report.json",
                "native-dispatch-plan.json",
                "native-dispatch-inputs.json",
            ):
                (root / name).write_text("stale\n", encoding="utf-8")

        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                "FAKE_GH_LOG": str(log),
                "FAKE_UPSTREAM_REF": upstream_ref,
                "FAKE_UPSTREAM_COMMIT": upstream_commit,
                "FAKE_NATIVE_RELEASE_EXISTS": "true" if native_release_exists else "false",
                "FAKE_NATIVE_RELEASE_EXISTS_AFTER_PLAN": (
                    ""
                    if native_release_exists_after_plan is None
                    else ("true" if native_release_exists_after_plan else "false")
                ),
                "FAKE_NATIVE_RUN_STATUSES": json.dumps(native_run_statuses),
                "FAKE_NATIVE_RUN_STATUSES_AFTER_PLAN": (
                    ""
                    if native_run_statuses_after_plan is None
                    else json.dumps(native_run_statuses_after_plan)
                ),
                "FAKE_UPSTREAM_COMMIT_AFTER_PLAN": upstream_commit_after_plan,
                "FAKE_LATEST_RELEASE_ERROR": latest_release_error,
                "FAKE_COMMIT_LOOKUP_ERROR": commit_lookup_error,
                "FAKE_COMMIT_LOOKUP_ERROR_AFTER_PLAN": commit_lookup_error_after_plan,
                "FAKE_RELEASE_LOOKUP_ERROR": release_lookup_error,
                "FAKE_RELEASE_LOOKUP_ERROR_AFTER_PLAN": release_lookup_error_after_plan,
                "FAKE_RUNS_LOOKUP_ERROR": runs_lookup_error,
                "FAKE_RUNS_LOOKUP_ERROR_AFTER_PLAN": runs_lookup_error_after_plan,
                "FAKE_DISPATCH_ERROR": dispatch_error,
                "GH_TOKEN": "fake-token",
                "GITHUB_REPOSITORY": "leehack/llamadart-native",
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_RUN_ID": "424242",
                "GITHUB_SHA": NATIVE_HEAD_COMMIT,
                "GITHUB_REF_NAME": "main",
                "GITHUB_OUTPUT": str(output),
                "RUNNER_TEMP": str(root),
            }
        )
        process = subprocess.run(
            ["bash", "-c", self.discovery_block],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode == 0:
            outputs = dict(
                line.split("=", 1)
                for line in output.read_text().splitlines()
                if "=" in line
            )
            if outputs["decision"] == "dispatch":
                if mutate_dispatch_input is not None:
                    inputs_path = root / "native-dispatch-inputs.json"
                    inputs = json.loads(inputs_path.read_text())
                    inputs[mutate_dispatch_input[0]] = mutate_dispatch_input[1]
                    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
                process = subprocess.run(
                    ["bash", "-c", self.dispatch_block],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            elif outputs["decision"] == "fail":
                process = subprocess.run(
                    ["bash", "-c", self.failure_block],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
        return OrchestrationRun(process, root, log)

    def test_candidate_dispatches_exactly_one_exact_native_release(self) -> None:
        run = self.run_orchestration()
        self.assertEqual(0, run.process.returncode, run.process.stderr)
        self.assertEqual(1, len(run.dispatches), run.gh_calls)
        argv = run.dispatches[0]["argv"]
        self.assertEqual(
            [
                "workflow",
                "run",
                "native_release.yml",
                "--repo",
                "leehack/llamadart-native",
                "--ref",
                "main",
                "--json",
            ],
            argv,
        )
        self.assertEqual(
            {
                "llama_cpp_tag": "v9.9.9",
                "llama_cpp_commit": UPSTREAM_COMMIT,
                "native_release_tag": "v9.9.9",
                "smoke_policy": "required",
                "correlation_id": deterministic_correlation_id(
                    upstream_ref="v9.9.9",
                    upstream_commit=UPSTREAM_COMMIT,
                    native_release_tag="v9.9.9",
                ),
                "publish_release": "true",
            },
            run.dispatch_inputs,
        )
        self.assertEqual("true", run.dispatch_inputs["publish_release"])
        self.assertIs(True, run.report["dispatch"]["inputs"]["publish_release"])
        self.assertEqual("candidate", run.report["status"])
        self.assertEqual("dispatch", run.report["decision"])
        self.assertEqual("candidate", run.outputs["status"])
        self.assertEqual("dispatch", run.outputs["decision"])

    def test_existing_native_release_dispatches_nothing(self) -> None:
        run = self.run_orchestration(native_release_exists=True)
        self.assertEqual(0, run.process.returncode, run.process.stderr)
        self.assertEqual([], run.dispatches)
        self.assertEqual("noop", run.report["status"])
        self.assertEqual("skip", run.report["decision"])

    def test_incompatible_upstream_fails_closed_without_dispatch(self) -> None:
        run = self.run_orchestration(upstream_ref="b10545")
        self.assertNotEqual(0, run.process.returncode)
        self.assertEqual([], run.dispatches)
        self.assertEqual("incompatible", run.report["status"])
        self.assertEqual("fail", run.report["decision"])
        self.assertIn("stable vMAJOR.MINOR.PATCH", run.report["message"])

    def test_queued_and_in_progress_runs_suppress_duplicate_dispatch(self) -> None:
        for statuses in (("queued",), ("in_progress",), ("queued", "in_progress")):
            with self.subTest(statuses=statuses):
                run = self.run_orchestration(native_run_statuses=statuses)
                self.assertEqual(0, run.process.returncode, run.process.stderr)
                self.assertEqual([], run.dispatches)
                self.assertEqual("candidate", run.report["status"])
                self.assertEqual("skip", run.report["decision"])
                self.assertEqual(len(statuses), run.report["in_flight_native_runs"])

    def test_state_changes_after_evidence_suppress_or_fail_before_dispatch(self) -> None:
        cases = (
            {"native_release_exists_after_plan": True, "returncode": 0},
            {"native_run_statuses_after_plan": ("queued",), "returncode": 0},
            {"native_run_statuses_after_plan": ("in_progress",), "returncode": 0},
            {"upstream_commit_after_plan": "c" * 40, "returncode": 1},
            {
                "commit_lookup_error_after_plan": "gh: API unavailable (HTTP 503)",
                "returncode": 1,
            },
            {
                "release_lookup_error_after_plan": "gh: API unavailable (HTTP 503)",
                "returncode": 1,
            },
            {
                "runs_lookup_error_after_plan": "gh: API unavailable (HTTP 503)",
                "returncode": 1,
            },
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                expected_returncode = arguments["returncode"]
                parameters = {
                    key: value for key, value in arguments.items() if key != "returncode"
                }
                run = self.run_orchestration(**parameters)
                self.assertEqual(expected_returncode, run.process.returncode)
                self.assertEqual([], run.dispatches)
                self.assertEqual("dispatch", run.report["decision"])

    def test_api_failure_cannot_expose_seeded_stale_evidence(self) -> None:
        run = self.run_orchestration(
            latest_release_error="gh: API unavailable (HTTP 503)",
            seed_stale_outputs=True,
        )
        self.assertNotEqual(0, run.process.returncode)
        self.assertEqual([], run.dispatches)
        self.assertIsNone(run.report)
        self.assertIsNone(run.plan)
        self.assertFalse((run.root / "native-dispatch-inputs.json").exists())

    def test_dispatch_transport_must_match_the_uploaded_plan(self) -> None:
        run = self.run_orchestration(
            mutate_dispatch_input=("correlation_id", "tampered-correlation")
        )
        self.assertNotEqual(0, run.process.returncode)
        self.assertEqual([], run.dispatches)
        self.assertIn("does not match uploaded evidence", run.process.stderr)

    def test_repeated_detection_reuses_the_same_correlation_id(self) -> None:
        first = self.run_orchestration()
        second = self.run_orchestration()
        self.assertEqual(
            first.dispatch_inputs["correlation_id"],
            second.dispatch_inputs["correlation_id"],
        )

    def test_dispatch_never_publishes_or_mutates_the_repository(self) -> None:
        workflow = AUTO_WORKFLOW.read_text()
        for forbidden in (
            "gh release",
            "git push",
            "git commit",
            "git tag",
            "submodule update",
            "submodules:",
            "third_party",
            "contents: write",
        ):
            self.assertNotIn(forbidden, workflow)
        self.assertEqual(1, workflow.count("actions: write"))
        self.assertEqual(1, workflow.count("gh workflow run"))

    def test_in_flight_query_counts_every_unsettled_native_release_run(self) -> None:
        if shutil.which("jq") is None:
            self.skipTest("jq is unavailable")
        program = re.search(
            r"\| jq -r '(\[\.\[\]\.workflow_runs\[\].*?)'",
            self.discovery_block,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(program, self.discovery_block)
        payload = [
            {
                "workflow_runs": [
                    {"status": "queued"},
                    {"status": "in_progress"},
                    {"status": "waiting"},
                    {"status": "requested"},
                    {"status": "pending"},
                    {"status": "completed"},
                ]
            },
            {
                "workflow_runs": [
                    {"status": "completed"},
                ]
            },
        ]
        result = subprocess.run(
            ["jq", program.group(1)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.assertEqual(5, int(result.stdout.strip()))

        settled = [
            {"workflow_runs": [{"status": "completed"}]},
            {"workflow_runs": [{"status": "completed"}]},
        ]
        result = subprocess.run(
            ["jq", program.group(1)],
            input=json.dumps(settled),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.assertEqual(0, int(result.stdout.strip()))

    def test_unreadable_in_flight_count_fails_closed_without_dispatch(self) -> None:
        run = self.run_orchestration(runs_lookup_error="gh: API unavailable (HTTP 503)")
        self.assertNotEqual(0, run.process.returncode)
        self.assertEqual([], run.dispatches)
        self.assertIsNone(run.report)

    def test_upstream_api_and_commit_errors_fail_closed_without_dispatch(self) -> None:
        cases = (
            {"latest_release_error": "gh: API unavailable (HTTP 503)"},
            {"commit_lookup_error": "gh: API unavailable (HTTP 503)"},
            {"upstream_commit": "abc"},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                run = self.run_orchestration(**arguments)
                self.assertNotEqual(0, run.process.returncode)
                self.assertEqual([], run.dispatches)
                self.assertIsNone(run.report)

    def test_release_lookup_errors_are_not_misclassified_as_absence(self) -> None:
        for error in (
            "gh: access denied (HTTP 403)",
            "gh: Not Found",
            "gh: API unavailable (HTTP 503)",
        ):
            with self.subTest(error=error):
                run = self.run_orchestration(release_lookup_error=error)
                self.assertNotEqual(0, run.process.returncode)
                self.assertEqual([], run.dispatches)
                self.assertIsNone(run.report)

    def test_dispatch_failure_preserves_the_planned_evidence(self) -> None:
        run = self.run_orchestration(dispatch_error="gh: dispatch rejected (HTTP 422)")
        self.assertNotEqual(0, run.process.returncode)
        self.assertEqual(1, len(run.dispatches))
        self.assertEqual("candidate", run.report["status"])
        self.assertEqual("dispatch", run.report["decision"])
        self.assertEqual("true", run.dispatch_inputs["publish_release"])


if __name__ == "__main__":
    unittest.main()
