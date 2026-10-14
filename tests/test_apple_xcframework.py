from __future__ import annotations

from pathlib import Path
import plistlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import apple_bundle_metadata as metadata  # noqa: E402
import package_apple_xcframework as packager  # noqa: E402
import validate_apple_xcframework as validator  # noqa: E402


VALIDATOR = ROOT / "tools/validate_apple_xcframework.py"
SHIPPED_MANIFEST = packager.PRIVACY_MANIFEST.read_bytes()
IOS = "ios-arm64"
SIMULATOR = "ios-arm64_x86_64-simulator"
MACOS = "macos-arm64_x86_64"
ROOT_PLIST = "llama.xcframework/Info.plist"
IOS_MANIFEST = f"llama.xcframework/{IOS}/llama.framework/PrivacyInfo.xcprivacy"
MACOS_MANIFEST = (
    f"llama.xcframework/{MACOS}/llama.framework/Versions/A/Resources/"
    "PrivacyInfo.xcprivacy"
)
IOS_INFO = f"llama.xcframework/{IOS}/llama.framework/Info.plist"
SIMULATOR_MANIFEST = (
    f"llama.xcframework/{SIMULATOR}/llama.framework/PrivacyInfo.xcprivacy"
)
SIMULATOR_INFO = f"llama.xcframework/{SIMULATOR}/llama.framework/Info.plist"
MACOS_INFO = f"llama.xcframework/{MACOS}/llama.framework/Versions/A/Resources/Info.plist"
IOS_BINARY_MEMBER = f"llama.xcframework/{IOS}/llama.framework/llama"
SIMULATOR_BINARY_MEMBER = f"llama.xcframework/{SIMULATOR}/llama.framework/llama"
MACOS_BINARY_MEMBER = f"llama.xcframework/{MACOS}/llama.framework/Versions/A/llama"
MACOS_PLATFORM, IOS_PLATFORM, IOS_SIMULATOR_PLATFORM = 1, 2, 7


def encode_version(version: str) -> int:
    major, minor, patch = (list(map(int, version.split("."))) + [0, 0])[:3]
    return major << 16 | minor << 8 | patch


def thin_macho(platform: int | None, minos: str = "16.4") -> bytes:
    """Return a minimal 64-bit Mach-O header with one LC_BUILD_VERSION (or none)."""
    commands = b""
    if platform is not None:
        commands = struct.pack(
            "<6I",
            metadata.LC_BUILD_VERSION,
            24,
            platform,
            encode_version(minos),
            encode_version("26.5"),
            0,
        )
    header = struct.pack(
        "<8I", metadata.MH_MAGIC_64, 0x0100000C, 0, 6, int(bool(commands)),
        len(commands), 0, 0,
    )
    return header + commands


def fat_macho(*images: bytes, wide: bool = False) -> bytes:
    """Return a FAT container, or FAT_64 (64-bit offsets and sizes) when ``wide``."""
    magic, layout, stride = (
        (metadata.FAT_MAGIC_64, ">iiQQII", 32)
        if wide
        else (metadata.FAT_MAGIC, ">iiIII", 20)
    )
    offset = 8 + stride * len(images)
    entries, bodies = b"", b""
    for image in images:
        fields = [0x0100000C, 0, offset + len(bodies), len(image), 0]
        entries += struct.pack(layout, *fields, *([0] if wide else []))
        bodies += image
    return struct.pack(">II", magic, len(images)) + entries + bodies


IOS_BINARY = thin_macho(IOS_PLATFORM, "16.4")
SIMULATOR_BINARY = fat_macho(
    thin_macho(IOS_SIMULATOR_PLATFORM, "16.4"),
    thin_macho(IOS_SIMULATOR_PLATFORM, "16.4"),
)
MACOS_BINARY = fat_macho(
    thin_macho(MACOS_PLATFORM, "13.3"), thin_macho(MACOS_PLATFORM, "13.3")
)


def info_plist(binary: bytes, **overrides: object) -> bytes:
    content: dict[str, object] = {
        "CFBundleExecutable": "llama",
        "CFBundleIdentifier": packager.BUNDLE_IDENTIFIER,
        "CFBundlePackageType": "FMWK",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        **metadata.framework_os_metadata(binary),
    }
    content.update(overrides)
    return plistlib.dumps({k: v for k, v in content.items() if v is not None})


def manifest(**overrides: object) -> bytes:
    content = plistlib.loads(SHIPPED_MANIFEST)
    content.update(overrides)
    return plistlib.dumps(content)


def accessed(category: object, *reasons: object) -> dict[str, object]:
    return {
        "NSPrivacyAccessedAPIType": category,
        "NSPrivacyAccessedAPITypeReasons": list(reasons),
    }


def root_plist(ios_slice_platform: str = "ios", simulator: bool = True) -> bytes:
    libraries: list[dict[str, str]] = [
        {
            "LibraryIdentifier": IOS,
            "LibraryPath": "llama.framework",
            "BinaryPath": "llama.framework/llama",
            "SupportedPlatform": ios_slice_platform,
        },
        {
            "LibraryIdentifier": MACOS,
            "LibraryPath": "llama.framework",
            "BinaryPath": "llama.framework/Versions/A/llama",
            "SupportedPlatform": "macos",
        },
    ]
    if simulator:
        libraries.insert(
            1,
            {
                "LibraryIdentifier": SIMULATOR,
                "LibraryPath": "llama.framework",
                "BinaryPath": "llama.framework/llama",
                "SupportedPlatform": "ios",
                "SupportedPlatformVariant": "simulator",
            },
        )
    return plistlib.dumps({"AvailableLibraries": libraries})


def write_archive(directory: str, members: dict[str, bytes | None]) -> Path:
    """Write a three-slice XCFramework zip; a None value drops that member."""
    content: dict[str, bytes | None] = {
        ROOT_PLIST: root_plist(),
        IOS_BINARY_MEMBER: IOS_BINARY,
        IOS_INFO: info_plist(IOS_BINARY),
        IOS_MANIFEST: SHIPPED_MANIFEST,
        SIMULATOR_BINARY_MEMBER: SIMULATOR_BINARY,
        SIMULATOR_INFO: info_plist(SIMULATOR_BINARY),
        SIMULATOR_MANIFEST: SHIPPED_MANIFEST,
        MACOS_BINARY_MEMBER: MACOS_BINARY,
        MACOS_INFO: info_plist(MACOS_BINARY),
        MACOS_MANIFEST: SHIPPED_MANIFEST,
    }
    content.update(members)
    archive = Path(directory) / "llama.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in content.items():
            if data is not None:
                bundle.writestr(name, data)
    return archive


class ShippedManifestTests(unittest.TestCase):
    def test_declares_only_the_audited_file_timestamp_reason(self) -> None:
        errors, declared = validator.validate_manifest(SHIPPED_MANIFEST)

        self.assertEqual([], errors)
        self.assertEqual({validator.FILE_TIMESTAMP}, declared)
        self.assertEqual(
            [accessed(validator.FILE_TIMESTAMP, "C617.1")],
            plistlib.loads(SHIPPED_MANIFEST)["NSPrivacyAccessedAPITypes"],
        )


class PackagerTests(unittest.TestCase):
    def test_packaged_slices_carry_the_manifest_where_the_validator_expects_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            ios_source = work / "ios.dylib"
            ios_source.write_bytes(IOS_BINARY)
            simulator_source = work / "simulator.dylib"
            simulator_source.write_bytes(SIMULATOR_BINARY)
            macos_source = work / "macos.dylib"
            macos_source.write_bytes(MACOS_BINARY)
            xcframework = work / "llama.xcframework"
            with patch.object(packager, "run"):
                ios = packager.make_ios_framework(
                    ios_source, xcframework / IOS / "llama.framework"
                )
                packager.make_ios_framework(
                    simulator_source, xcframework / SIMULATOR / "llama.framework"
                )
                macos = packager.make_macos_framework(
                    macos_source, xcframework / MACOS / "llama.framework"
                )
            (xcframework / "Info.plist").write_bytes(root_plist())
            archive = work / "llama.zip"
            packager.zip_xcframework(xcframework, archive)

            self.assertEqual(
                SHIPPED_MANIFEST, (ios / "PrivacyInfo.xcprivacy").read_bytes()
            )
            self.assertEqual(
                SHIPPED_MANIFEST,
                (macos / "Versions/A/Resources/PrivacyInfo.xcprivacy").read_bytes(),
            )
            self.assertFalse((macos / "PrivacyInfo.xcprivacy").exists())
            self.assertEqual([], validator.validate_archive(archive))

    def test_framework_info_plists_declare_the_binary_platform_and_minimum(
        self,
    ) -> None:
        cases = {
            "device": (
                IOS_BINARY,
                packager.make_ios_framework,
                "Info.plist",
                {"CFBundleSupportedPlatforms": ["iPhoneOS"], "MinimumOSVersion": "16.4"},
            ),
            "simulator": (
                SIMULATOR_BINARY,
                packager.make_ios_framework,
                "Info.plist",
                {
                    "CFBundleSupportedPlatforms": ["iPhoneSimulator"],
                    "MinimumOSVersion": "16.4",
                },
            ),
            "macos": (
                MACOS_BINARY,
                packager.make_macos_framework,
                "Versions/A/Resources/Info.plist",
                {
                    "CFBundleSupportedPlatforms": ["MacOSX"],
                    "LSMinimumSystemVersion": "13.3",
                },
            ),
        }
        for name, (binary, make, plist_path, expected) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "libllamadart.dylib"
                source.write_bytes(binary)
                with patch.object(packager, "run"):
                    framework = make(source, Path(directory) / "llama.framework")
                info = plistlib.loads((framework / plist_path).read_bytes())

                self.assertEqual(expected, {key: info.get(key) for key in expected})
                self.assertEqual("FMWK", info["CFBundlePackageType"])

    def test_refuses_a_binary_whose_architectures_disagree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "libllamadart.dylib"
            source.write_bytes(
                fat_macho(
                    thin_macho(IOS_SIMULATOR_PLATFORM, "16.4"),
                    thin_macho(IOS_SIMULATOR_PLATFORM, "17.0"),
                )
            )
            with patch.object(packager, "run"), self.assertRaisesRegex(
                ValueError, "architectures disagree"
            ):
                packager.make_ios_framework(source, Path(directory) / "llama.framework")


class ValidateArchiveTests(unittest.TestCase):
    def errors(self, members: dict[str, bytes | None]) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            return validator.validate_archive(write_archive(directory, members))

    def test_accepts_a_manifest_in_every_slice(self) -> None:
        self.assertEqual([], self.errors({}))

    def test_rejects_a_slice_without_a_manifest(self) -> None:
        self.assertEqual(
            [f"{IOS}: missing privacy manifest at {IOS_MANIFEST}"],
            self.errors({IOS_MANIFEST: None}),
        )

    def test_rejects_a_manifest_outside_the_versioned_resources_directory(self) -> None:
        stray = f"llama.xcframework/{MACOS}/llama.framework/PrivacyInfo.xcprivacy"

        self.assertEqual(
            [
                f"{MACOS}: unexpected privacy manifest at {stray}",
                f"{MACOS}: missing privacy manifest at {MACOS_MANIFEST}",
            ],
            self.errors({MACOS_MANIFEST: None, stray: SHIPPED_MANIFEST}),
        )

    def test_rejects_a_manifest_that_is_not_a_property_list(self) -> None:
        errors = self.errors({MACOS_MANIFEST: b"<plist><dict>"})

        self.assertEqual(1, len(errors))
        self.assertIn(f"{MACOS_MANIFEST} is not a valid property list", errors[0])

    def test_rejects_inaccurate_declarations(self) -> None:
        cases = {
            "NSPrivacyTracking must be false": manifest(NSPrivacyTracking=True),
            "NSPrivacyCollectedDataTypes must be an empty array": manifest(
                NSPrivacyCollectedDataTypes=[{"NSPrivacyCollectedDataType": "x"}]
            ),
            "NSPrivacyAccessedAPITypes must be an array": manifest(
                NSPrivacyAccessedAPITypes="C617.1"
            ),
            "unknown required-reason API category: 'Timestamp'": manifest(
                NSPrivacyAccessedAPITypes=[accessed("Timestamp", "C617.1")]
            ),
            f"{validator.FILE_TIMESTAMP} declares unapproved reason '35F9.1'": manifest(
                NSPrivacyAccessedAPITypes=[accessed(validator.FILE_TIMESTAMP, "35F9.1")]
            ),
            f"{validator.FILE_TIMESTAMP} must declare at least one reason": manifest(
                NSPrivacyAccessedAPITypes=[accessed(validator.FILE_TIMESTAMP)]
            ),
            "unknown required-reason API category: ['Timestamp']": manifest(
                NSPrivacyAccessedAPITypes=[accessed(["Timestamp"], "C617.1")]
            ),
            f"{validator.FILE_TIMESTAMP} declares unapproved reason ['C617.1']": manifest(
                NSPrivacyAccessedAPITypes=[accessed(validator.FILE_TIMESTAMP, ["C617.1"])]
            ),
        }
        for expected, data in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    [f"{IOS}: {IOS_MANIFEST} {expected}"],
                    self.errors({IOS_MANIFEST: data}),
                )

    def test_import_audit_requires_declarations_to_match_each_binary(self) -> None:
        references = {
            IOS_BINARY: ({"_stat", "_fstat", "_mach_absolute_time"}, set()),
            SIMULATOR_BINARY: ({"_stat"}, set()),
            MACOS_BINARY: ({"_abort"}, set()),
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            validator,
            "binary_api_references",
            side_effect=lambda binary: references[binary.read_bytes()],
        ):
            errors = validator.validate_archive(
                write_archive(directory, {}), audit_imports=True
            )

        self.assertEqual(
            [
                f"{IOS}: uses {validator.SYSTEM_BOOT_TIME} via _mach_absolute_time "
                "but the manifest does not declare it",
                f"{MACOS}: manifest declares {validator.FILE_TIMESTAMP} but the "
                "binary uses none of its APIs",
            ],
            errors,
        )

    def test_rejects_framework_info_plists_that_misdescribe_the_binary(self) -> None:
        cases = {
            "missing MinimumOSVersion": (
                {IOS_INFO: info_plist(IOS_BINARY, MinimumOSVersion=None)},
                [f"{IOS}: {IOS_INFO} lacks MinimumOSVersion (binary requires '16.4')"],
            ),
            "MinimumOSVersion below the binary": (
                {IOS_INFO: info_plist(IOS_BINARY, MinimumOSVersion="13.0")},
                [
                    f"{IOS}: {IOS_INFO} MinimumOSVersion is '13.0' but the "
                    "binary requires '16.4'"
                ],
            ),
            "missing CFBundleSupportedPlatforms": (
                {IOS_INFO: info_plist(IOS_BINARY, CFBundleSupportedPlatforms=None)},
                [
                    f"{IOS}: {IOS_INFO} lacks CFBundleSupportedPlatforms "
                    "(binary requires ['iPhoneOS'])"
                ],
            ),
            "missing LSMinimumSystemVersion": (
                {MACOS_INFO: info_plist(MACOS_BINARY, LSMinimumSystemVersion=None)},
                [
                    f"{MACOS}: {MACOS_INFO} lacks LSMinimumSystemVersion "
                    "(binary requires '13.3')"
                ],
            ),
            "missing CFBundleVersion": (
                {IOS_INFO: info_plist(IOS_BINARY, CFBundleVersion=None)},
                [f"{IOS}: {IOS_INFO} lacks CFBundleVersion"],
            ),
            "simulator binary in the device slice": (
                {
                    IOS_BINARY_MEMBER: thin_macho(IOS_SIMULATOR_PLATFORM),
                    IOS_INFO: info_plist(IOS_BINARY),
                },
                [
                    f"{IOS}: {IOS_BINARY_MEMBER} targets iPhoneSimulator but the "
                    "slice is iPhoneOS",
                    f"{IOS}: {IOS_INFO} CFBundleSupportedPlatforms is ['iPhoneOS'] "
                    "but the binary requires ['iPhoneSimulator']",
                ],
            ),
            "device platform in the simulator slice Info.plist": (
                {
                    SIMULATOR_INFO: info_plist(
                        SIMULATOR_BINARY, CFBundleSupportedPlatforms=["iPhoneOS"]
                    )
                },
                [
                    f"{SIMULATOR}: {SIMULATOR_INFO} CFBundleSupportedPlatforms is "
                    "['iPhoneOS'] but the binary requires ['iPhoneSimulator']"
                ],
            ),
            "device binary in the simulator slice": (
                {
                    SIMULATOR_BINARY_MEMBER: IOS_BINARY,
                    SIMULATOR_INFO: info_plist(SIMULATOR_BINARY),
                },
                [
                    f"{SIMULATOR}: {SIMULATOR_BINARY_MEMBER} targets iPhoneOS but "
                    "the slice is iPhoneSimulator",
                    f"{SIMULATOR}: {SIMULATOR_INFO} CFBundleSupportedPlatforms is "
                    "['iPhoneSimulator'] but the binary requires ['iPhoneOS']",
                ],
            ),
            "binary without a build version": (
                {IOS_BINARY_MEMBER: thin_macho(None)},
                [
                    f"{IOS}: {IOS_BINARY_MEMBER}: binary has no LC_BUILD_VERSION "
                    "load command"
                ],
            ),
            "missing Info.plist": (
                {IOS_INFO: None},
                [f"{IOS}: missing framework Info.plist at {IOS_INFO}"],
            ),
            "Info.plist outside the versioned resources directory": (
                {
                    MACOS_INFO: None,
                    f"llama.xcframework/{MACOS}/llama.framework/Info.plist": info_plist(
                        MACOS_BINARY
                    ),
                },
                [f"{MACOS}: missing framework Info.plist at {MACOS_INFO}"],
            ),
        }
        for name, (members, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(expected, self.errors(members))

    def test_cli_fails_on_an_unreadable_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "llama.zip"
            archive.write_bytes(b"not a zip")
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(archive)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(1, result.returncode)
        self.assertIn("error:", result.stderr)


@unittest.skipUnless(
    sys.platform == "darwin",
    "needs the Apple toolchain to build and inspect Mach-O binaries",
)
class MachOImportAuditTests(unittest.TestCase):
    def test_metadata_matches_the_toolchain_build_version(self) -> None:
        targets = {
            "arm64-apple-ios16.4": (
                "iphoneos",
                {"CFBundleSupportedPlatforms": ["iPhoneOS"], "MinimumOSVersion": "16.4"},
            ),
            "x86_64-apple-ios16.4-simulator": (
                "iphonesimulator",
                {
                    "CFBundleSupportedPlatforms": ["iPhoneSimulator"],
                    "MinimumOSVersion": "16.4",
                },
            ),
            "arm64-apple-macos13.3": (
                "macosx",
                {
                    "CFBundleSupportedPlatforms": ["MacOSX"],
                    "LSMinimumSystemVersion": "13.3",
                },
            ),
        }
        for target, (sdk, expected) in targets.items():
            with self.subTest(target), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "plain.c"
                source.write_text(self.PLAIN, encoding="utf-8")
                binary = Path(directory) / "plain.dylib"
                subprocess.run(
                    [
                        "xcrun", "--sdk", sdk, "clang", "-dynamiclib", "-target",
                        target, str(source), "-o", str(binary),
                    ],
                    check=True,
                    capture_output=True,
                )

                self.assertEqual(
                    expected, metadata.framework_os_metadata(binary.read_bytes())
                )

    PLAIN = "int plain(void) { return 1; }\n"
    REQUIRED_REASON = (
        "#import <Foundation/Foundation.h>\n"
        "#include <mach/mach_time.h>\n"
        "#include <sys/stat.h>\n"
        "double required_reason(int fd) {\n"
        "  struct stat info;\n"
        '  fstat(fd, &info);\n'
        '  stat("/", &info);\n'
        "  return [[NSProcessInfo processInfo] systemUptime] + mach_absolute_time();\n"
        "}\n"
    )

    def fat_dylib(self, work: Path, sources: dict[str, str]) -> bytes:
        thin = []
        for arch, source in sources.items():
            source_file = work / f"{arch}.m"
            source_file.write_text(source, encoding="utf-8")
            thin.append(work / f"{arch}.dylib")
            subprocess.run(
                [
                    "xcrun", "clang", "-dynamiclib", "-arch", arch,
                    "-framework", "Foundation", str(source_file), "-o", str(thin[-1]),
                ],
                check=True,
                capture_output=True,
            )
        fat = work / "fat.dylib"
        subprocess.run(
            ["xcrun", "lipo", "-create", *map(str, thin), "-output", str(fat)],
            check=True,
            capture_output=True,
        )
        return fat.read_bytes()

    def test_audits_every_architecture_of_a_fat_slice(self) -> None:
        architectures = ("arm64", "x86_64")
        for importing in architectures:
            with self.subTest(importing=importing), tempfile.TemporaryDirectory() as directory:
                binary = self.fat_dylib(
                    Path(directory),
                    {
                        arch: self.REQUIRED_REASON if arch == importing else self.PLAIN
                        for arch in architectures
                    },
                )
                # Both slices hold the same host-built macOS dylib.
                archive = write_archive(
                    directory,
                    {
                        ROOT_PLIST: root_plist(
                            ios_slice_platform="macos", simulator=False
                        ),
                        IOS_BINARY_MEMBER: binary,
                        IOS_INFO: info_plist(binary),
                        MACOS_BINARY_MEMBER: binary,
                        MACOS_INFO: info_plist(binary),
                    },
                )

                self.assertEqual(
                    [
                        f"{identifier}: uses {validator.SYSTEM_BOOT_TIME} via "
                        "_mach_absolute_time, systemUptime but the manifest does "
                        "not declare it"
                        for identifier in (IOS, MACOS)
                    ],
                    validator.validate_archive(archive, audit_imports=True),
                )


class FrameworkOsMetadataTests(unittest.TestCase):
    def test_reads_the_platform_and_minimum_of_thin_and_fat_binaries(self) -> None:
        self.assertEqual(
            {"CFBundleSupportedPlatforms": ["iPhoneOS"], "MinimumOSVersion": "16.4"},
            metadata.framework_os_metadata(IOS_BINARY),
        )
        self.assertEqual(
            {"CFBundleSupportedPlatforms": ["MacOSX"], "LSMinimumSystemVersion": "13.3"},
            metadata.framework_os_metadata(MACOS_BINARY),
        )
        self.assertEqual(
            {
                "CFBundleSupportedPlatforms": ["iPhoneSimulator"],
                "MinimumOSVersion": "16.4",
            },
            metadata.framework_os_metadata(SIMULATOR_BINARY),
        )
        self.assertEqual(
            "16.4.1",
            metadata.framework_os_metadata(thin_macho(IOS_PLATFORM, "16.4.1"))[
                "MinimumOSVersion"
            ],
        )

    def test_reads_every_architecture_of_a_fat_64_container(self) -> None:
        fat_64 = fat_macho(
            thin_macho(IOS_SIMULATOR_PLATFORM, "17.2"),
            thin_macho(IOS_SIMULATOR_PLATFORM, "17.2"),
            wide=True,
        )

        self.assertEqual(
            [(IOS_SIMULATOR_PLATFORM, encode_version("17.2"))] * 2,
            metadata.build_versions(fat_64),
        )
        self.assertEqual(
            {
                "CFBundleSupportedPlatforms": ["iPhoneSimulator"],
                "MinimumOSVersion": "17.2",
            },
            metadata.framework_os_metadata(fat_64),
        )

    def test_rejects_binaries_without_one_supported_build_version(self) -> None:
        cases = {
            "architectures disagree": fat_macho(
                thin_macho(IOS_SIMULATOR_PLATFORM, "16.4"),
                thin_macho(MACOS_PLATFORM, "16.4"),
            ),
            "architecture 1 has no LC_BUILD_VERSION": fat_macho(
                thin_macho(IOS_PLATFORM), thin_macho(None)
            ),
            "binary has no LC_BUILD_VERSION": thin_macho(None),
            "unsupported Mach-O platform 6": thin_macho(6),
            "not a little-endian Mach-O image": b"ios binary",
            "truncated Mach-O binary": IOS_BINARY[:40],
        }
        for expected, binary in cases.items():
            with self.subTest(expected), self.assertRaisesRegex(ValueError, expected):
                metadata.framework_os_metadata(binary)


class RequiredCategoryTests(unittest.TestCase):
    def test_maps_symbol_variants_classes_and_selectors_to_categories(self) -> None:
        self.assertEqual(
            {
                validator.FILE_TIMESTAMP: ["_fstat", "_stat"],
                validator.SYSTEM_BOOT_TIME: ["systemUptime"],
                validator.DISK_SPACE: ["_statfs"],
                validator.USER_DEFAULTS: ["_OBJC_CLASS_$_NSUserDefaults"],
            },
            validator.required_categories(
                {
                    "_stat$INODE64",
                    "_fstat",
                    "_statfs$INODE64",
                    "_OBJC_CLASS_$_NSUserDefaults",
                    "_clock_gettime",
                },
                {"systemUptime", "fileURLWithPath:"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
