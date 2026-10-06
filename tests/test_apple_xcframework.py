from __future__ import annotations

from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import package_apple_xcframework as packager  # noqa: E402
import validate_apple_xcframework as validator  # noqa: E402


VALIDATOR = ROOT / "tools/validate_apple_xcframework.py"
SHIPPED_MANIFEST = packager.PRIVACY_MANIFEST.read_bytes()
IOS = "ios-arm64"
MACOS = "macos-arm64_x86_64"
IOS_MANIFEST = f"llama.xcframework/{IOS}/llama.framework/PrivacyInfo.xcprivacy"
MACOS_MANIFEST = (
    f"llama.xcframework/{MACOS}/llama.framework/Versions/A/Resources/"
    "PrivacyInfo.xcprivacy"
)


def manifest(**overrides: object) -> bytes:
    content = plistlib.loads(SHIPPED_MANIFEST)
    content.update(overrides)
    return plistlib.dumps(content)


def accessed(category: str, *reasons: str) -> dict[str, object]:
    return {
        "NSPrivacyAccessedAPIType": category,
        "NSPrivacyAccessedAPITypeReasons": list(reasons),
    }


def write_archive(directory: str, members: dict[str, bytes | None]) -> Path:
    """Write a two-slice XCFramework zip; a None value drops that member."""
    content: dict[str, bytes | None] = {
        "llama.xcframework/Info.plist": plistlib.dumps(
            {
                "AvailableLibraries": [
                    {
                        "LibraryIdentifier": IOS,
                        "LibraryPath": "llama.framework",
                        "BinaryPath": "llama.framework/llama",
                    },
                    {
                        "LibraryIdentifier": MACOS,
                        "LibraryPath": "llama.framework",
                        "BinaryPath": "llama.framework/Versions/A/llama",
                    },
                ]
            }
        ),
        f"llama.xcframework/{IOS}/llama.framework/llama": b"ios",
        IOS_MANIFEST: SHIPPED_MANIFEST,
        f"llama.xcframework/{MACOS}/llama.framework/Versions/A/llama": b"macos",
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
            source = work / "libllamadart.dylib"
            source.write_bytes(b"binary")
            xcframework = work / "llama.xcframework"
            with patch.object(packager, "run"):
                ios = packager.make_ios_framework(
                    source, xcframework / IOS / "llama.framework"
                )
                macos = packager.make_macos_framework(
                    source, xcframework / MACOS / "llama.framework"
                )
            with (xcframework / "Info.plist").open("wb") as file:
                plistlib.dump(
                    {
                        "AvailableLibraries": [
                            {"LibraryIdentifier": IOS, "LibraryPath": ios.name},
                            {"LibraryIdentifier": MACOS, "LibraryPath": macos.name},
                        ]
                    },
                    file,
                )
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
        }
        for expected, data in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    [f"{IOS}: {IOS_MANIFEST} {expected}"],
                    self.errors({IOS_MANIFEST: data}),
                )

    def test_import_audit_requires_declarations_to_match_each_binary(self) -> None:
        references = {
            b"ios": ({"_stat", "_fstat", "_mach_absolute_time"}, set()),
            b"macos": ({"_abort"}, set()),
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
