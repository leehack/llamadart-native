from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build import ANDROID_ARM64_CPU_VARIANTS, android_arm64_cpu_variant_libs  # noqa: E402
from ci_scope import native_required  # noqa: E402


WORKFLOW = ROOT / ".github/workflows/native_release.yml"
BUILD_PY = ROOT / "tools/build.py"
THIS_TEST = "tests/test_native_release_dedup.py"
SETUP_CCACHE = ROOT / ".github/actions/setup-ccache/action.yml"
CCACHE_STATS = ROOT / ".github/actions/ccache-stats/action.yml"
APT_RETRY = ROOT / "tools/apt_retry.sh"
SOURCES_TEMPLATE = ROOT / "tools/docker/ubuntu.sources.template"
DOCKERFILE = ROOT / "tools/docker/linux-builder.Dockerfile"

CODENAME_PLACEHOLDER = "@VERSION_CODENAME@"
SED_RENDER = f's/{CODENAME_PLACEHOLDER}/${{VERSION_CODENAME}}/g'

CCACHE_LANES = {
    "build-android": {
        "key-prefix": "ccache-android-${{ matrix.android_abi }}-${{ matrix.backend }}",
        "extra-restore-keys": "ccache-android-${{ matrix.android_abi }}-",
        "label": "android/${{ matrix.android_abi }}/${{ matrix.backend }}",
    },
    "build-apple": {
        "key-prefix": "ccache-apple-${{ matrix.target }}",
        "extra-restore-keys": None,
        "label": "apple/${{ matrix.target }}",
    },
    "build-linux": {
        "key-prefix": "ccache-linux-${{ matrix.arch }}-${{ matrix.backend }}",
        "extra-restore-keys": "ccache-linux-${{ matrix.arch }}-",
        "label": "linux/${{ matrix.arch }}/${{ matrix.backend }}",
    },
    "build-linux-hip": {
        "key-prefix": "ccache-linux-x64-hip",
        "extra-restore-keys": None,
        "label": "linux/x64/hip",
    },
}

HIP_APT_INSTALL = (
    "DEBIAN_FRONTEND=noninteractive apt_get_install build-essential binutils ccache "
    "ninja-build pkg-config python3 python3-pip rocblas-dev hipblas-dev"
)


def workflow_steps(text: str, name: str) -> list[str]:
    found = []
    marker = f"      - name: {name}\n"
    start = text.find(marker)
    while start != -1:
        body = text[start + 1 :]
        end = body.find("\n      - ")
        found.append(body if end == -1 else body[: end + 1])
        start = text.find(marker, start + 1)
    return found


def workflow_jobs(text: str) -> dict[str, str]:
    body = text[text.index("\njobs:\n") :]
    starts = [(m.group(1), m.start()) for m in re.finditer(r"\n  ([a-z][\w-]*):\n", body)]
    return {
        name: body[pos : starts[i + 1][1] if i + 1 < len(starts) else len(body)]
        for i, (name, pos) in enumerate(starts)
    }


def steps_by_job(text: str, name: str) -> dict[str, list[str]]:
    found = {}
    for job, block in workflow_jobs(text).items():
        steps = workflow_steps(block, name)
        if steps:
            found[job] = steps
    return found


def step_input(step: str, key: str) -> str | None:
    match = re.search(rf"^ *{re.escape(key)}: (.+)$", step, re.MULTILINE)
    return match.group(1) if match else None

LIST_COMMAND = "python3 tools/build.py list --android-arm64-cpu-variant-libs"

SINGLED_OUT_VARIANT_LIBS = frozenset(
    {
        "libggml-cpu-android_armv8.2_2.so",
        "libggml-cpu-android_armv8.0_1.so",
    }
)

EXPECTED_VARIANT_LIBS = frozenset(
    {
        "libggml-cpu-android_armv8.0_1.so",
        "libggml-cpu-android_armv8.2_1.so",
        "libggml-cpu-android_armv8.2_2.so",
        "libggml-cpu-android_armv8.6_1.so",
        "libggml-cpu-android_armv9.0_1.so",
        "libggml-cpu-android_armv9.2_1.so",
        "libggml-cpu-android_armv9.2_2.so",
    }
)


class CpuVariantListingTests(unittest.TestCase):
    def _run_list(self, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/build.py"), "list", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        return result.stdout

    def test_option_prints_one_library_per_declared_variant(self) -> None:
        lines = self._run_list("--android-arm64-cpu-variant-libs").splitlines()
        expected = [f"libggml-cpu-{name}.so" for name, _arch, _features in ANDROID_ARM64_CPU_VARIANTS]
        self.assertEqual(expected, lines)
        self.assertEqual(len(set(lines)), len(lines))

    def test_listed_libraries_are_the_seven_variants_the_release_ships(self) -> None:
        lines = self._run_list("--android-arm64-cpu-variant-libs").splitlines()
        self.assertEqual(EXPECTED_VARIANT_LIBS, set(lines))
        self.assertEqual(len(EXPECTED_VARIANT_LIBS), len(lines))
        self.assertLessEqual(SINGLED_OUT_VARIANT_LIBS, EXPECTED_VARIANT_LIBS)

    def test_bare_list_still_prints_presets(self) -> None:
        stdout = self._run_list()
        self.assertIn("android: abi=arm64-v8a", stdout)
        self.assertNotIn("libggml-cpu-", stdout)

    def test_android_builder_names_its_outputs_through_the_helper(self) -> None:
        body = BUILD_PY.read_text().split("def build_android_arm64_abi(")[1].split("\ndef ")[0]
        self.assertIn("android_arm64_cpu_variant_libs()", body)
        self.assertNotIn("libggml-cpu-", body)


class CiScopeTests(unittest.TestCase):
    def test_editing_this_file_alone_does_not_require_the_native_matrix(self) -> None:
        self.assertTrue((ROOT / THIS_TEST).is_file())
        self.assertFalse(native_required([THIS_TEST]))
        self.assertTrue(native_required([THIS_TEST, "tools/build.py"]))


class NativeReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text()

    def test_validation_step_reads_the_variant_list_from_build_py(self) -> None:
        self.assertIn(f'variant_list="$({LIST_COMMAND})"', self.workflow)
        self.assertIn('mapfile -t expected_cpu_variants <<< "$variant_list"', self.workflow)

    def test_empty_variant_list_fails_the_step(self) -> None:
        self.assertIn('if [ -z "$variant_list" ]; then', self.workflow)
        self.assertIn('echo "tools/build.py listed no Android arm64 CPU variants" >&2', self.workflow)

    def test_every_listed_variant_library_must_exist(self) -> None:
        self.assertIn('for lib in "${expected_cpu_variants[@]}"; do', self.workflow)
        self.assertIn('test -f "$OUT_DIR/$lib" || { echo "Missing $OUT_DIR/$lib"; exit 1; }', self.workflow)

    def test_workflow_does_not_restate_the_variant_list(self) -> None:
        mentioned = set(re.findall(r"libggml-cpu-android_[\w.]+\.so", self.workflow))
        self.assertLessEqual(mentioned, SINGLED_OUT_VARIANT_LIBS)

    def test_singled_out_variant_libraries_are_really_built(self) -> None:
        self.assertLessEqual(SINGLED_OUT_VARIANT_LIBS, set(android_arm64_cpu_variant_libs()))

    def test_no_cuda_version_literal_survives(self) -> None:
        self.assertEqual([], re.findall(r"release \d+\\?\.\d+", self.workflow))
        self.assertEqual([], re.findall(r"CUDA\\v\d", self.workflow))

    def test_nvcc_checks_derive_their_pattern_from_the_env_var(self) -> None:
        self.assertIn(
            'cuda_major_minor="$(printf \'%s\' "${CUDA_REDIST_VERSION}" | cut -d. -f1,2)"',
            self.workflow,
        )
        self.assertIn(
            'grep -Eq "release ${cuda_major_minor//./\\\\.}([^0-9.]|$)" nvcc-version.txt',
            self.workflow,
        )
        self.assertIn(
            "$cudaMajorMinor = (($env:CUDA_REDIST_VERSION -split '\\.') | Select-Object -First 2) -join '.'",
            self.workflow,
        )
        self.assertIn(
            "$cudaVersionPattern = 'release ' + [regex]::Escape($cudaMajorMinor) + '([^0-9.]|$)'",
            self.workflow,
        )
        self.assertIn("-notmatch $cudaVersionPattern", self.workflow)

    def test_windows_cuda_root_derives_from_the_env_var(self) -> None:
        self.assertIn(
            'Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\\CUDA\\v$cudaMajorMinor"',
            self.workflow,
        )


class CcacheCompositeActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text()
        self.setup_action = SETUP_CCACHE.read_text()
        self.stats_action = CCACHE_STATS.read_text()

    def test_cache_key_and_restore_keys_derive_from_the_key_prefix(self) -> None:
        self.assertIn("path: ${{ inputs.cache-dir }}", self.setup_action)
        self.assertIn("key: ${{ inputs.key-prefix }}-${{ github.run_id }}", self.setup_action)
        self.assertIn(
            "restore-keys: |\n"
            "          ${{ inputs.key-prefix }}-\n"
            "          ${{ inputs.extra-restore-keys }}\n",
            self.setup_action,
        )

    def test_priming_still_sizes_and_zeroes_the_cache(self) -> None:
        self.assertIn("CCACHE_DIR: ${{ inputs.cache-dir }}", self.setup_action)
        self.assertIn("CCACHE_MAXSIZE: ${{ inputs.max-size }}", self.setup_action)
        self.assertIn(
            'mkdir -p "${CCACHE_DIR}"\n'
            '        ccache --max-size "${CCACHE_MAXSIZE}"\n'
            "        ccache --zero-stats\n",
            self.setup_action,
        )

    def test_summary_heading_comes_from_the_label_input(self) -> None:
        self.assertIn("CCACHE_STATS_LABEL: ${{ inputs.label }}", self.stats_action)
        self.assertIn('echo "### ccache stats (${CCACHE_STATS_LABEL})"', self.stats_action)
        self.assertEqual(2, self.stats_action.count("ccache --show-stats || true"))

    def test_every_lane_passes_the_cache_key_it_used_before(self) -> None:
        setups = steps_by_job(self.workflow, "Set up ccache")
        self.assertEqual(set(CCACHE_LANES), set(setups))
        for job, lane in CCACHE_LANES.items():
            self.assertEqual(1, len(setups[job]), job)
            step = setups[job][0]
            self.assertIn("uses: ./.github/actions/setup-ccache", step)
            self.assertEqual("${{ env.CCACHE_DIR }}", step_input(step, "cache-dir"), job)
            self.assertEqual("${{ env.CCACHE_MAXSIZE }}", step_input(step, "max-size"), job)
            self.assertEqual(lane["key-prefix"], step_input(step, "key-prefix"), job)
            self.assertEqual(
                lane["extra-restore-keys"], step_input(step, "extra-restore-keys"), job
            )

    def test_every_lane_reports_stats_under_its_old_heading(self) -> None:
        reports = steps_by_job(self.workflow, "Report ccache stats")
        self.assertEqual(set(CCACHE_LANES), set(reports))
        for job, lane in CCACHE_LANES.items():
            self.assertEqual(1, len(reports[job]), job)
            step = reports[job][0]
            self.assertIn("if: always()", step)
            self.assertIn("uses: ./.github/actions/ccache-stats", step)
            self.assertEqual(lane["label"], step_input(step, "label"), job)

    def test_no_lane_keeps_its_own_copy_of_the_scaffolding(self) -> None:
        self.assertNotIn("ccache --zero-stats", self.workflow)
        self.assertNotIn("ccache --show-stats", self.workflow)
        self.assertNotIn("actions/cache@v6", self.workflow)
        self.assertNotIn("CCACHE_MAXSIZE}", self.workflow)

    def test_windows_keeps_its_own_sccache_setup(self) -> None:
        self.assertIn("uses: mozilla-actions/sccache-action@v0.0.11", self.workflow)
        self.assertIn("SCCACHE_GHA_ENABLED=true", self.workflow)
        self.assertNotIn("sccache", self.setup_action)


class AptRetryScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text()
        self.script = APT_RETRY.read_text()

    def test_script_keeps_the_retry_ladder_and_apt_options(self) -> None:
        self.assertIn("apt-get -o Acquire::Retries=3 \"$@\"", self.script)
        self.assertIn("for attempt in 1 2 3; do", self.script)
        self.assertIn("sleep $((attempt * 15))", self.script)
        self.assertIn("_apt_get_retry update", self.script)
        self.assertIn('_apt_get_retry install -y "$@"', self.script)

    def test_script_uses_sudo_only_when_not_root(self) -> None:
        self.assertIn('if [ "$(id -u)" -eq 0 ]; then', self.script)
        self.assertIn("sudo apt-get -o Acquire::Retries=3", self.script)

    def test_post_checkout_steps_source_the_script(self) -> None:
        self.assertEqual(2, self.workflow.count(". tools/apt_retry.sh"))

    def test_only_the_pre_checkout_step_still_declares_the_helpers(self) -> None:
        self.assertEqual(1, self.workflow.count("apt_get_install() {"))
        self.assertEqual(1, self.workflow.count("apt_get_update() {"))
        install_git = workflow_steps(self.workflow, "Install Git")
        self.assertEqual(1, len(install_git))
        self.assertIn("apt_get_install() {", install_git[0])
        self.assertLess(
            self.workflow.index("apt_get_install() {"),
            self.workflow.index("uses: ./.github/actions/checkout-llama-ref\n", self.workflow.index("build-linux-hip:")),
        )

    def test_container_step_installs_noninteractively_without_exporting_it(self) -> None:
        hip_deps = steps_by_job(self.workflow, "Install build deps")["build-linux-hip"][0]
        self.assertEqual(
            [HIP_APT_INSTALL],
            [line.strip() for line in hip_deps.splitlines() if "DEBIAN_FRONTEND" in line],
        )
        self.assertIn("python3 -m pip install --upgrade cmake", hip_deps)

    def test_the_hosted_linux_lane_sets_no_apt_frontend(self) -> None:
        linux_deps = steps_by_job(self.workflow, "Install build deps")["build-linux"][0]
        self.assertNotIn("DEBIAN_FRONTEND", linux_deps)
        self.assertNotIn("DEBIAN_FRONTEND", APT_RETRY.read_text())


class UbuntuSourcesTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text()
        self.template = SOURCES_TEMPLATE.read_text()
        self.dockerfile = DOCKERFILE.read_text()

    def render(self, codename: str) -> str:
        return self.template.replace(CODENAME_PLACEHOLDER, codename)

    def test_workflow_and_dockerfile_render_the_same_template(self) -> None:
        self.assertFalse((ROOT / "tools/docker/ubuntu.sources").exists())
        self.assertIn(f'sed "{SED_RENDER}" tools/docker/ubuntu.sources.template', self.workflow)
        self.assertIn(f'sed "{SED_RENDER}" /etc/apt/ubuntu.sources.template', self.dockerfile)
        self.assertIn(". /etc/os-release", self.dockerfile)
        self.assertIn("source /etc/os-release", self.workflow)
        self.assertIn("COPY ubuntu.sources.template /etc/apt/ubuntu.sources.template", self.dockerfile)

    def test_no_codename_is_hardcoded(self) -> None:
        self.assertNotIn("noble", self.template)
        self.assertNotIn("noble", self.workflow)
        self.assertNotIn("Signed-By", self.workflow)

    def test_rendered_list_keeps_all_three_suites(self) -> None:
        rendered = self.render("plucky")
        self.assertNotIn("@", rendered)
        self.assertEqual(
            [
                "Suites: plucky plucky-updates plucky-backports",
                "Suites: plucky-security",
                "Suites: plucky plucky-updates plucky-backports plucky-security",
            ],
            [line for line in rendered.splitlines() if line.startswith("Suites:")],
        )
        self.assertEqual(
            ["Architectures: amd64", "Architectures: amd64", "Architectures: arm64"],
            [line for line in rendered.splitlines() if line.startswith("Architectures:")],
        )
        self.assertEqual(
            [
                "URIs: http://archive.ubuntu.com/ubuntu",
                "URIs: http://security.ubuntu.com/ubuntu",
                "URIs: http://ports.ubuntu.com/ubuntu-ports",
            ],
            [line for line in rendered.splitlines() if line.startswith("URIs:")],
        )


if __name__ == "__main__":
    unittest.main()
