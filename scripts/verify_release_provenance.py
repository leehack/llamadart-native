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
RELEASE_ASSETS = {
    f"llamadart-native-{bundle}-b10545.tar.gz": (*meta, "core", "core")
    for bundle, meta in RUNTIME_BUNDLES.items()
}
RELEASE_ASSETS.update(
    {
        "llamadart-native-apple-xcframework-b10545.zip": (
            "apple",
            "universal",
            "core",
            "spm-xcframework",
        ),
        "llamadart-native-headers-b10545.tar.gz": (
            "all",
            "universal",
            "core",
            "headers",
        ),
    }
)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def verify_workflow_contract(errors: list[str]) -> None:
    workflow = WORKFLOW.read_text()
    wrapper_workflow = WRAPPER_WORKFLOW.read_text()
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
    linux_validation = workflow.find(
        "for archive in release_assets/llamadart-native-linux-*.tar.gz"
    )
    linux_archive_glob = workflow.find('archives=("$dir"/*.tar.gz)')
    linux_archive_nullglob = workflow.rfind(
        "shopt -s nullglob", 0, linux_archive_glob
    )
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
        "linux-artifact-contract:" in wrapper_workflow
        and "arch: [x64, arm64]" in wrapper_workflow
        and "--backend cpu" in wrapper_workflow
        and "qemu-aarch64" in wrapper_workflow
        and "tools/linux_dlopen_smoke.c" in wrapper_workflow,
        "PR validation must inspect and clean-dlopen Linux x64 and arm64 archives",
        errors,
    )


def verify_manifest_contract(errors: list[str]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = root / "assets"
        assets.mkdir()
        expected_payloads = {}
        for index, filename in enumerate(RELEASE_ASSETS, start=1):
            payload = f"native-test-{index}\n".encode()
            (assets / filename).write_bytes(payload)
            expected_payloads[filename] = payload
        output_json = root / "assets.json"
        output_checksums = root / "SHA256SUMS"
        env = os.environ.copy()
        env.update(
            {
                "LLAMADART_LLAMA_CPP_TAG": "b-test",
                "LLAMADART_LLAMA_CPP_COMMIT": "1" * 40,
                "LLAMADART_NATIVE_COMMIT": "2" * 40,
            }
        )
        subprocess.run(
            [
                str(MANIFEST_SCRIPT),
                "b10545",
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
            manifest.get("llama_cpp_tag") == "b-test"
            and manifest.get("llama_cpp_commit") == "1" * 40
            and manifest.get("native_commit") == "2" * 40,
            "assets.json must contain the upstream ref, exact upstream commit, and native release commit",
            errors,
        )
        require(
            set(manifest)
            == {
                "tag",
                "llama_cpp_tag",
                "llama_cpp_commit",
                "native_commit",
                "generated_at",
                "hook_contract_version",
                "artifacts",
            }
            and manifest.get("tag") == "b10545"
            and manifest.get("hook_contract_version") == 1
            and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                manifest.get("generated_at", ""),
            )
            is not None,
            "assets.json must preserve the release manifest schema",
            errors,
        )

        artifacts = manifest.get("artifacts", [])
        require(
            [artifact.get("file") for artifact in artifacts]
            == sorted(RELEASE_ASSETS),
            "assets.json must contain every supported release asset in filename order",
            errors,
        )
        for artifact in artifacts:
            filename = artifact.get("file")
            expected_meta = RELEASE_ASSETS.get(filename)
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
                f"{filename}: artifact entry must preserve the manifest schema",
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
                f"{filename}: incorrect platform/arch/backend/module classification",
                errors,
            )
            require(
                artifact.get("sha256") == hashlib.sha256(payload).hexdigest()
                and artifact.get("size") == len(payload),
                f"{filename}: incorrect checksum or size",
                errors,
            )

        expected_checksums = "".join(
            f"{hashlib.sha256(expected_payloads[filename]).hexdigest()}  {filename}\n"
            for filename in sorted(RELEASE_ASSETS)
        )
        require(
            output_checksums.read_text() == expected_checksums,
            "SHA256SUMS must contain the exact checksum for every supported release asset",
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
