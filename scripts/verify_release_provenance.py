#!/usr/bin/env python3
"""Verify native release source provenance contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/native_release.yml"
CHECKOUT_ACTION = ROOT / ".github/actions/checkout-llama-ref/action.yml"
MANIFEST_SCRIPT = ROOT / "scripts/generate_assets_manifest.sh"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def verify_workflow_contract(errors: list[str]) -> None:
    workflow = WORKFLOW.read_text()
    action = CHECKOUT_ACTION.read_text()

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

    experiment_marker = "cuda-prebuilt-experiment"
    require(
        workflow.count(
            "if: ${{ github.event.inputs.native_release_tag != "
            f"'{experiment_marker}' }}"
        )
        == 5,
        "the CUDA prebuilt experiment must skip every normal platform build job",
        errors,
    )
    require(
        f'if [ "$RELEASE_TAG" = "{experiment_marker}" ]' in workflow
        and "is strictly non-publishing" in workflow,
        "the CUDA prebuilt experiment marker must reject publishing dispatches",
        errors,
    )
    require(
        "windows-cuda-prebuilt-experiment:" in workflow
        and "github.event.inputs.publish_release == 'false'" in workflow
        and "Install exact GPU-less fatbin inspector" in workflow
        and "CUDA_CUOBJDUMP_ARCHIVE_SHA256" in workflow
        and "--cuobjdump $env:CUDA_CUOBJDUMP" in workflow
        and "python tools/package_upstream_cuda.py" in workflow
        and "python tools/verify_cuda_pack.py" in workflow
        and "tools/smoke_windows_cuda_pack.py" in workflow
        and "python @smokeArgs" in workflow
        and "compression-level: 0" in workflow,
        "the non-publishing Windows experiment must inspect fatbins, independently verify, loader-smoke, and upload precompressed CUDA packs",
        errors,
    )

    windows_build = workflow.split("  build-windows:", 1)[1].split(
        "  package-windows-cuda:", 1
    )[0]
    cuda_packaging = workflow.split("  package-windows-cuda:", 1)[1].split(
        "  windows-cuda-prebuilt-experiment:", 1
    )[0]
    require(
        "backend: cuda" not in windows_build
        and "Install CUDA Toolkit" not in windows_build,
        "Windows release builds must not compile the replaced CUDA backend lane",
        errors,
    )
    require(
        "needs: [resolve-tag, build-windows]" in cuda_packaging
        and "name: native_windows_x64_vulkan" in cuda_packaging
        and "--native-release-tag $env:NATIVE_RELEASE_TAG" in cuda_packaging
        and "--core-dll $env:CUDA_PREBUILT_CORE_DLL" in cuda_packaging
        and "name: windows_cuda_packs" in cuda_packaging,
        "production CUDA sidecars must be verified against and follow the same-run Windows core",
        errors,
    )
    require(
        "package-windows-cuda]" in workflow
        and 'cuda_pack_dir="artifacts/windows_cuda_packs"' in workflow
        and 'test "$cuda_pack_count" = "2"' in workflow,
        "release packaging must require and publish both verified Windows CUDA sidecars",
        errors,
    )


def verify_manifest_contract(errors: list[str]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = root / "assets"
        assets.mkdir()
        (assets / "libllamadart-linux-x64.so").write_bytes(b"native-test")
        (assets / "llamadart-native-windows-x64-cuda13-native-test.tar.gz").write_bytes(
            b"cuda-test"
        )
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
                "native-test",
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
            "libllamadart-linux-x64.so" in output_checksums.read_text(),
            "manifest generation must continue to emit asset checksums",
            errors,
        )
        cuda_artifacts = [
            artifact
            for artifact in manifest.get("artifacts", [])
            if artifact.get("file")
            == "llamadart-native-windows-x64-cuda13-native-test.tar.gz"
        ]
        require(
            len(cuda_artifacts) == 1
            and cuda_artifacts[0].get("module") == "backend-cuda13"
            and cuda_artifacts[0].get("platform") == "windows"
            and cuda_artifacts[0].get("arch") == "x64"
            and cuda_artifacts[0].get("backend") == "cuda"
            and cuda_artifacts[0].get("size") == len(b"cuda-test"),
            "assets.json must classify Windows CUDA sidecars as versioned CUDA backends",
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
