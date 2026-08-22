#!/usr/bin/env python3
"""Verify native release source provenance contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/native_release.yml"
WRAPPER_WORKFLOW = ROOT / ".github/workflows/validate_wrapper.yml"
AUTO_WORKFLOW = ROOT / ".github/workflows/auto_native_release.yml"
CHECKOUT_ACTION = ROOT / ".github/actions/checkout-llama-ref/action.yml"
MANIFEST_SCRIPT = ROOT / "scripts/generate_assets_manifest.sh"
VALIDATION_WORKFLOW = ROOT / ".github/workflows/validate_release_provenance.yml"
POLICY_DOC = ROOT / "docs/release_version_policy.md"
README = ROOT / "README.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
PUBLICATION_SCRIPT = ROOT / "scripts/release_publication.py"
CONTRACT_SCRIPT = ROOT / "scripts/release_contract.py"
SMOKE_SCRIPT = ROOT / "tools/smoke_linux_bundle.py"

RUNTIME_BUNDLES = {
    "android-arm64": ("android", "arm64"),
    "android-x64": ("android", "x64"),
    "ios-arm64": ("ios", "arm64"),
    "ios-arm64-sim": ("ios", "arm64-sim"),
    "ios-x86_64-sim": ("ios", "x86_64-sim"),
    "linux-arm64": ("linux", "arm64"),
    "linux-x64": ("linux", "x64"),
    "macos-arm64": ("macos", "arm64"),
    "macos-x86_64": ("macos", "x86_64"),
    "windows-arm64": ("windows", "arm64"),
    "windows-x64": ("windows", "x64"),
}


def release_assets(tag: str) -> dict[str, tuple[str, str, str, str]]:
    assets = {
        f"llamadart-native-{bundle}-{tag}.tar.gz": (*meta, "core", "core")
        for bundle, meta in RUNTIME_BUNDLES.items()
    }
    assets.update(
        {
            f"llamadart-native-apple-xcframework-{tag}.zip": (
                "apple",
                "universal",
                "core",
                "spm-xcframework",
            ),
            f"llamadart-native-headers-{tag}.tar.gz": (
                "all",
                "universal",
                "core",
                "headers",
            ),
        }
    )
    return assets


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def workflow_job(workflow: str, name: str) -> str:
    match = re.search(
        rf"^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else ""


def verify_workflow_contract(errors: list[str]) -> None:
    workflow = WORKFLOW.read_text()
    wrapper_workflow = WRAPPER_WORKFLOW.read_text()
    auto_workflow = AUTO_WORKFLOW.read_text()
    validation_workflow = VALIDATION_WORKFLOW.read_text()
    publication_script = PUBLICATION_SCRIPT.read_text()
    contract_script = CONTRACT_SCRIPT.read_text()
    smoke_script = SMOKE_SCRIPT.read_text()
    action = CHECKOUT_ACTION.read_text()

    workflow_bundles = set(
        re.findall(r'^\s+bundle="([^"]+)"', workflow, flags=re.MULTILINE)
    )
    for template, expanded in {
        "windows-${BASH_REMATCH[1]}": ("windows-x64", "windows-arm64"),
        "linux-${BASH_REMATCH[1]}": ("linux-x64", "linux-arm64"),
    }.items():
        if template in workflow_bundles:
            workflow_bundles.remove(template)
            workflow_bundles.update(expanded)
    require(
        workflow_bundles == set(RUNTIME_BUNDLES),
        "manifest fixture must cover every runtime bundle emitted by the release workflow",
        errors,
    )

    expected_commit_input = (
        "expected_commit: ${{ needs.resolve-tag.outputs.llama_cpp_commit }}"
    )
    require(
        "llama_cpp_commit: ${{ steps.llama-checkout.outputs.commit }}" in workflow,
        "resolve-tag must expose the exact llama.cpp commit",
        errors,
    )
    require(
        "expected_commit: ${{ steps.contract.outputs.llama_cpp_commit }}" in workflow,
        "initial checkout must enforce the normalized exact commit from the dispatch contract",
        errors,
    )
    require(
        workflow.count(expected_commit_input) == 7,
        "all six build/package checkouts and the submodule update must assert the resolved commit",
        errors,
    )
    require(
        "llama_cpp_commit:" in workflow
        and "correlation_id:" in workflow
        and "smoke_policy:" in workflow
        and "scripts/release_contract.py validate-dispatch" in workflow
        and "latest" not in workflow[: workflow.find("permissions:")]
        and "submodule" not in workflow[: workflow.find("permissions:")],
        "release dispatch must require exact ref/commit, native tag, smoke policy, and correlation",
        errors,
    )
    require(
        "LLAMADART_LLAMA_CPP_COMMIT: ${{ needs.resolve-tag.outputs.llama_cpp_commit }}"
        in workflow
        and "LLAMADART_NATIVE_COMMIT: ${{ steps.provenance.outputs.native_commit }}"
        in workflow,
        "manifest generation must record resolved upstream and release provenance commits",
        errors,
    )
    package_job = workflow_job(workflow, "package-and-release")
    publish_job = workflow_job(workflow, "publish-release")
    update_job = workflow_job(workflow, "update-llama-submodule")
    provenance = package_job.find("- name: Create release provenance commit")
    manifest = package_job.find("- name: Generate manifest + checksums")
    bundle = package_job.find("- name: Bundle immutable publication inputs")
    upload = package_job.find("- name: Upload verified publication transaction")
    require(
        -1 not in (provenance, manifest, bundle, upload)
        and provenance < manifest < bundle < upload,
        "release provenance must be committed, manifested, and bundled before "
        "publication input upload",
        errors,
    )
    require(
        'tree_commit="$(git ls-tree HEAD third_party/llama.cpp | awk \'{print $3}\')"'
        in package_job
        and "release-provenance.bundle" in package_job
        and 'tree_commit="$(git ls-tree "$EXPECTED_NATIVE_COMMIT" third_party/llama.cpp'
        in publish_job,
        "the publication transaction must carry and revalidate the provenance commit tree",
        errors,
    )

    read_only_jobs = (
        "resolve-tag",
        "build-android",
        "build-apple",
        "build-linux",
        "build-linux-hip",
        "build-windows",
        "smoke-release-contract",
        "package-and-release",
        "emit-release-result",
    )
    require(
        re.search(r"^permissions:\n  contents: read$", workflow, re.MULTILINE)
        is not None
        and workflow.count("contents: write") == 2,
        "workflow default must be contents:read with exactly two narrow write jobs",
        errors,
    )
    for job_name in read_only_jobs:
        job = workflow_job(workflow, job_name)
        require(bool(job), f"release workflow must define {job_name}", errors)
        require(
            "permissions:" not in job
            and job.count("uses: actions/checkout@v7")
            == job.count("persist-credentials: false"),
            f"{job_name} must inherit read-only permissions and must not persist "
            "checkout credentials",
            errors,
        )
    require(
        "contents: write" in publish_job
        and "persist-credentials: true" in publish_job
        and "scripts/release_publication.py" in publish_job,
        "only the final publication job may retain credentials needed to create the release",
        errors,
    )
    require(
        "contents: write" in update_job
        and "persist-credentials: true" in update_job
        and "needs: [resolve-tag, publish-release]" in update_job,
        "stable submodule update must remain a separate, post-publication narrow write job",
        errors,
    )
    require(
        "git push" not in package_job
        and "gh release" not in package_job
        and "contents: write" not in package_job,
        "candidate packaging must not contain publication or repository writes",
        errors,
    )

    require(
        "expected_commit:" in action
        and 'if [ -n "$EXPECTED_COMMIT" ] && [ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]'
        in action
        and 'echo "commit=$ACTUAL_COMMIT" >> "$GITHUB_OUTPUT"' in action,
        "checkout action must expose and enforce exact commit provenance",
        errors,
    )
    linux_validation = package_job.find(
        "for archive in release_assets/llamadart-native-linux-*.tar.gz"
    )
    linux_archive_glob = package_job.find('archives=("$dir"/*.tar.gz)')
    linux_archive_nullglob = package_job.rfind(
        "shopt -s nullglob", 0, linux_archive_glob
    )
    linux_artifact_job = workflow_job(wrapper_workflow, "linux-artifact-contract")
    require(
        "python3 tools/package_linux_artifact.py" in workflow
        and workflow.count("python3 tools/validate_linux_artifact.py") == 2
        and 'tar -xzf "${archives[0]}" -C "$out_dir"' in workflow
        and -1 < linux_validation < manifest,
        "Linux release packaging must preserve and validate SONAME symlinks before manifest generation",
        errors,
    )
    require(
        -1 < linux_archive_nullglob < linux_archive_glob,
        "Linux release archive discovery must use nullglob so an empty directory reports zero archives",
        errors,
    )
    require(
        bool(linux_artifact_job)
        and "arch: [x64, arm64]" in linux_artifact_job
        and "--backend cpu" in linux_artifact_job
        and "qemu-aarch64" in linux_artifact_job
        and "tools/linux_dlopen_smoke.c" in linux_artifact_job
        and "persist-credentials: false" in linux_artifact_job,
        "PR validation must inspect and clean-dlopen Linux x64 and arm64 archives",
        errors,
    )
    require(
        "remote set-url" not in action
        and "credential.helper" not in action
        and "git config" not in action,
        "external-source checkout must not persist its read credential in git configuration",
        errors,
    )
    require(
        "scripts/release_version_policy.py" in workflow
        and "--existing-tags-file" in workflow
        and "--allow-existing-candidate" in workflow
        and "github_prerelease: ${{ steps.contract.outputs.github_prerelease }}" in workflow
        and '--prerelease "${{ needs.resolve-tag.outputs.github_prerelease }}"'
        in publish_job,
        "native release workflow must enforce version history while permitting "
        "exact retry reconciliation",
        errors,
    )
    require(
        "softprops/action-gh-release" not in workflow
        and "--clobber" not in workflow
        and "release-publication-${{ github.run_id }}" in package_job
        and "release-publication-${{ github.run_id }}" in publish_job
        and "publication_artifact_digest" in workflow
        and "${{ github.run_id }}" in publish_job,
        "publication must use the same immutable workflow transaction without "
        "mutable asset replacement",
        errors,
    )
    require(
        '"--draft"' in publication_script
        and '"--draft=false"' in publication_script
        and '"--clobber"' not in publication_script
        and '"--force"' not in publication_script
        and '"-a"' in publication_script
        and "llamadart-native publication transaction:" in publication_script
        and "tag {desired.tag!r} transaction mismatch" in publication_script
        and "native commit must be a full 40-hex SHA" in publication_script
        and "capture_output=True" in publication_script
        and "immutable asset mismatch" in publication_script
        and "release body/correlation mismatch" in publication_script,
        "publication driver must bind its annotated tag, remain draft-first and "
        "immutable, and fail closed on transaction, correlation, or digest mismatch",
        errors,
    )
    require(
        "needs: [resolve-tag, build-android, build-apple, build-linux, "
        "build-linux-hip, build-windows, smoke-release-contract]"
        in package_job
        and "needs: [resolve-tag, smoke-release-contract, package-and-release]" in publish_job
        and "needs.package-and-release.result == 'success'" in publish_job,
        "publication must remain downstream of the complete build and packaging matrix",
        errors,
    )
    smoke_job = workflow_job(workflow, "smoke-release-contract")
    result_job = workflow_job(workflow, "emit-release-result")
    require(
        "native_linux_x64_vulkan" in smoke_job
        and "tools/smoke_linux_bundle.py" in smoke_job
        and "llama_dart_tts_api_version" in smoke_script
        and "conclusion=passed" in smoke_job,
        "required publication smoke must load the packaged Linux wrapper and report its conclusion",
        errors,
    )
    require(
        "native-release-result-${{ github.run_id }}" in result_job
        and "scripts/release_contract.py release-result" in result_job
        and "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in result_job
        and "publication_artifact_digest" in result_job
        and "bundle_coverage" in contract_script
        and '"native_release_tag"' in contract_script
        and '"tag"' in contract_script,
        "release workflow must return exact correlated metadata, digests, bundle coverage, and aliases",
        errors,
    )
    require(
        "verify_historical_release_metadata.py --live" in validation_workflow
        and "persist-credentials: false" in validation_workflow
        and "python3 -m unittest discover" in validation_workflow,
        "validation CI must test historical live metadata, all policy tests, and "
        "non-persisted checkout credentials",
        errors,
    )
    resolve_tag = workflow.find("- name: Validate exact approved dispatch")
    resolve_commit = workflow.find("- name: Resolve exact llama.cpp commit")
    require(
        -1 not in (resolve_tag, resolve_commit)
        and "set -euo pipefail" in workflow[resolve_tag:resolve_commit],
        "release history collection must fail closed if any pipeline command fails",
        errors,
    )
    require(
        "scripts/release_version_policy.py" in auto_workflow
        and "--require-stable-upstream" in auto_workflow
        and "repos/ggml-org/llama.cpp/releases/latest" in auto_workflow,
        "automatic discovery must fail closed on anything except the upstream stable channel",
        errors,
    )
    require(
        auto_workflow.count("uses: actions/checkout@v7") > 0
        and auto_workflow.count("uses: actions/checkout@v7")
        == auto_workflow.count("persist-credentials: false"),
        "read-only candidate detection must not persist checkout credentials",
        errors,
    )
    require(
        "gh workflow run" not in auto_workflow
        and "publish_release=true" not in auto_workflow
        and "actions: write" not in auto_workflow
        and "native_release.yml" not in auto_workflow
        and "git push" not in auto_workflow
        and "git commit" not in auto_workflow
        and "submodule update" not in auto_workflow
        and "This detect-only workflow cannot dispatch, publish, or mutate" in auto_workflow,
        "scheduled automation must detect and prepare only, never dispatch publication",
        errors,
    )
    require(
        "native-discovery-report-${{ github.run_id }}" in auto_workflow
        and "scripts/release_contract.py discovery-report" in auto_workflow
        and 'status="candidate"' in auto_workflow
        and 'status="noop"' in auto_workflow
        and 'status="incompatible"' in auto_workflow
        and 'commits/${upstream_ref}' in auto_workflow,
        "scheduled discovery must emit exact candidate/noop/incompatible machine-readable evidence",
        errors,
    )
    require(
        "--require-stable-upstream 2>&1)" in auto_workflow
        and "2>&1 >/dev/null" not in auto_workflow,
        "incompatible discovery must retain the release-policy diagnostic in its JSON report",
        errors,
    )
    require(
        "schedule:" not in workflow[: workflow.find("permissions:")]
        and "workflow_dispatch:" in workflow[: workflow.find("permissions:")]
        and "publication requires smoke_policy=required" in contract_script,
        "only explicit exact dispatch may reach publication",
        errors,
    )
    require(
        "needs.resolve-tag.outputs.upstream_channel == 'stable'" in workflow
        and "needs.resolve-tag.outputs.release_kind == 'upstream'" in workflow,
        "nightly and wrapper-only releases must not move the repository's stable submodule pin",
        errors,
    )


def verify_documentation_contract(errors: list[str]) -> None:
    documentation = "\n".join(
        (POLICY_DOC.read_text(), README.read_text(), CONTRIBUTING.read_text())
    )
    require(
        "b10545" in documentation
        and "b10356-llamadart.1" in documentation
        and "prerelease=false" in documentation
        and "tag grammar" in documentation,
        "maintainer and consumer docs must preserve tag-driven historical channel semantics",
        errors,
    )
    require(
        "same workflow run" in documentation
        and "draft-first" in documentation
        and "fails closed" in documentation,
        "maintainer docs must explain exact-transaction partial-publication recovery",
        errors,
    )


def verify_manifest_contract(errors: list[str]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixtures = (
            ("stable", "v0.2.0", "v0.2.0"),
            ("nightly", "b10545", "b10545"),
            ("stable-wrapper", "v0.2.0-1", "v0.2.0"),
            ("nightly-wrapper", "b10545-1", "b10545"),
            ("legacy-wrapper", "b10356-llamadart.1", "b10356"),
        )
        for fixture_name, native_tag, upstream_ref in fixtures:
            fixture_root = root / fixture_name
            assets = fixture_root / "assets"
            assets.mkdir(parents=True)
            expected_assets = release_assets(native_tag)
            expected_payloads = {}
            for index, filename in enumerate(expected_assets, start=1):
                payload = f"{fixture_name}-native-test-{index}\n".encode()
                (assets / filename).write_bytes(payload)
                expected_payloads[filename] = payload
            output_json = fixture_root / "assets.json"
            output_checksums = fixture_root / "SHA256SUMS"
            env = os.environ.copy()
            env.update(
                {
                    "LLAMADART_LLAMA_CPP_TAG": upstream_ref,
                    "LLAMADART_LLAMA_CPP_COMMIT": "1" * 40,
                    "LLAMADART_NATIVE_COMMIT": "2" * 40,
                }
            )
            subprocess.run(
                [
                    str(MANIFEST_SCRIPT),
                    native_tag,
                    str(assets),
                    str(output_json),
                    str(output_checksums),
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )
            manifest = json.loads(output_json.read_text())
            require(
                manifest.get("native_release_tag") == native_tag
                and manifest.get("tag") == native_tag
                and manifest.get("llama_cpp_tag") == upstream_ref
                and manifest.get("llama_cpp_commit") == "1" * 40
                and manifest.get("native_commit") == "2" * 40,
                f"{fixture_name}: manifest must distinguish native tag, upstream ref, "
                "upstream commit, and native commit",
                errors,
            )
            require(
                set(manifest)
                == {
                    "tag",
                    "native_release_tag",
                    "llama_cpp_tag",
                    "llama_cpp_commit",
                    "native_commit",
                    "generated_at",
                    "hook_contract_version",
                    "artifacts",
                }
                and manifest.get("hook_contract_version") == 1
                and re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                    manifest.get("generated_at", ""),
                )
                is not None,
                f"{fixture_name}: assets.json must preserve the release manifest schema",
                errors,
            )

            artifacts = manifest.get("artifacts", [])
            require(
                [artifact.get("file") for artifact in artifacts]
                == sorted(expected_assets),
                f"{fixture_name}: manifest must contain every supported release asset",
                errors,
            )
            for artifact in artifacts:
                filename = artifact.get("file")
                expected_meta = expected_assets.get(filename)
                require(
                    set(artifact)
                    == {
                        "module",
                        "platform",
                        "arch",
                        "backend",
                        "file",
                        "sha256",
                        "size",
                    },
                    f"{fixture_name}/{filename}: artifact schema changed",
                    errors,
                )
                if expected_meta is None:
                    continue
                platform, arch, backend, module = expected_meta
                payload = expected_payloads[filename]
                require(
                    artifact.get("platform") == platform
                    and artifact.get("arch") == arch
                    and artifact.get("backend") == backend
                    and artifact.get("module") == module,
                    f"{fixture_name}/{filename}: incorrect artifact classification",
                    errors,
                )
                require(
                    artifact.get("sha256") == hashlib.sha256(payload).hexdigest()
                    and artifact.get("size") == len(payload),
                    f"{fixture_name}/{filename}: incorrect checksum or size",
                    errors,
                )

            expected_checksums = "".join(
                f"{hashlib.sha256(expected_payloads[filename]).hexdigest()}  {filename}\n"
                for filename in sorted(expected_assets)
            )
            require(
                output_checksums.read_text() == expected_checksums,
                f"{fixture_name}: SHA256SUMS must cover every release asset",
                errors,
            )


def main() -> int:
    errors: list[str] = []
    verify_workflow_contract(errors)
    verify_documentation_contract(errors)
    verify_manifest_contract(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Native release provenance contracts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
