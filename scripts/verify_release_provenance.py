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
AUTO_WORKFLOW = ROOT / ".github/workflows/auto_native_release.yml"
CHECKOUT_ACTION = ROOT / ".github/actions/checkout-llama-ref/action.yml"
MANIFEST_SCRIPT = ROOT / "scripts/generate_assets_manifest.sh"

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


def verify_workflow_contract(errors: list[str]) -> None:
    workflow = WORKFLOW.read_text()
    auto_workflow = AUTO_WORKFLOW.read_text()
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
        workflow.count(expected_commit_input) == 7,
        "all six build/package checkouts and the submodule update must assert the resolved commit",
        errors,
    )
    require(
        "if: ${{ github.event.inputs.llama_cpp_tag != 'submodule' }}" not in workflow,
        "submodule releases must use the same exact-commit checkout path",
        errors,
    )
    require(
        "llama_cpp_tag=submodule requires the pinned commit to have an exact "
        "vMAJOR.MINOR.PATCH or bNNNN tag" in workflow
        and "rev-parse --short HEAD" not in workflow,
        "untagged submodules must fail before release policy validation",
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
    provenance = workflow.find("- name: Create release provenance commit")
    manifest = workflow.find("- name: Generate manifest + checksums")
    verify_tag = workflow.find("- name: Verify release tag is still unused")
    push_tag = workflow.find("- name: Push release provenance tag")
    release = workflow.find("- name: Create release\n", push_tag)
    require(
        -1 not in (provenance, manifest, verify_tag, push_tag, release)
        and provenance < manifest < verify_tag < push_tag < release,
        "release provenance must be committed and manifested before the tag is pushed and released",
        errors,
    )
    require(
        'tree_commit="$(git ls-tree HEAD third_party/llama.cpp | awk \'{print $3}\')"'
        in workflow
        and 'git tag "$RELEASE_TAG" "$NATIVE_COMMIT"' in workflow,
        "the published tag must point to a commit whose tree records the resolved llama.cpp commit",
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
    require(
        "scripts/release_version_policy.py" in workflow
        and "--existing-tags-file" in workflow
        and "github_prerelease: ${{ steps.tag.outputs.github_prerelease }}" in workflow
        and "prerelease: ${{ needs.resolve-tag.outputs.github_prerelease }}" in workflow,
        "native release workflow must enforce version history and classify prereleases",
        errors,
    )
    resolve_tag = workflow.find("- name: Resolve llama.cpp tag")
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
        "gh workflow run" not in auto_workflow
        and "publish_release=true" not in auto_workflow
        and "actions: write" not in auto_workflow
        and "Publication requires explicit cross-repository approval" in auto_workflow,
        "scheduled automation must detect and prepare only, never dispatch publication",
        errors,
    )
    require(
        "needs.resolve-tag.outputs.upstream_channel == 'stable'" in workflow
        and "needs.resolve-tag.outputs.release_kind == 'upstream'" in workflow,
        "nightly and wrapper-only releases must not move the repository's stable submodule pin",
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
    verify_manifest_contract(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Native release provenance contracts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
