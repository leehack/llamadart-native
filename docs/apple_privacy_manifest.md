# Apple Privacy Manifest

Every framework slice of `llamadart-native-apple-xcframework-<tag>.zip` embeds
`tools/apple/PrivacyInfo.xcprivacy`. Apple requires an SDK to report its own
[required-reason API](https://developer.apple.com/documentation/bundleresources/describing-use-of-required-reason-api)
use; it cannot rely on the host app's manifest.

## Location

| Slice | Bundle layout | Manifest path inside `llama.framework` |
| --- | --- | --- |
| `ios-arm64`, `ios-arm64_x86_64-simulator` | flat | `PrivacyInfo.xcprivacy` |
| `macos-arm64_x86_64` | versioned | `Versions/A/Resources/PrivacyInfo.xcprivacy` |

A manifest in the root of a versioned framework is unsealed content and fails
code signing. The packager does not sign the frameworks; the consumer's Xcode
build seals the manifest when it embeds and signs them.

## Declarations

The SDK does not track, has no tracking domains, and collects no data.

| Category | Reason | Evidence |
| --- | --- | --- |
| `NSPrivacyAccessedAPICategoryFileTimestamp` | `C617.1` | Every slice and architecture imports `stat` and `fstat` (`stat$INODE64`/`fstat$INODE64` on macOS x86_64) and no other required-reason symbol or selector. |

The `stat`/`fstat` call sites in the consolidated Apple binary are:

- `fs_create_directory_with_parents` (`common/common.cpp`): checks whether the
  llama.cpp cache directory exists. Its only caller, `fs_get_cache_file`,
  resolves `LLAMA_CACHE` or `$HOME/Library/Caches/llama.cpp/`, which on iOS is
  inside the app container. Only llama.cpp's argument parser and model
  downloader call it; the wrapper uses neither.
- `httplib::detail::FileStat` and `httplib::detail::mmap::open` (vendored
  cpp-httplib): file type, size and modification time of the files its HTTP
  server serves. The wrapper never starts that server.

Neither path is reachable from the shipped C headers. Model, state and media
files are opened with `fopen`/`mmap`; this binary never passes them to `stat`
or `fstat`, so files a user grants through a document picker need no `3B52.1`
declaration. `C617.1` (metadata of files inside the app container) is the
reason that matches the remaining call sites. `0A2A.1` does not apply: the SDK
exposes no wrapper around a timestamp API.

`std::filesystem` and `clock_gettime` are not on Apple's list, and the binary
imports no boot-time, disk-space, user-defaults or active-keyboard API.

## Validation

`native_release.yml` runs this before generating the release manifest:

```bash
python3 tools/validate_apple_xcframework.py --audit-imports \
  release_assets/llamadart-native-apple-xcframework-<tag>.zip
```

It fails when a slice lacks the manifest, carries it in the wrong location,
ships an invalid property list or an unapproved reason, or when the declared
categories differ from the required-reason symbols and Objective-C selectors
the slice binary references (`nm -u`, `otool`). The audit cannot see calls
resolved through `dlsym`.

When an upstream bump makes the audit fail, find the new call sites, choose the
reason that describes them, and update the manifest and this page together.
Do not add a category only to make the check pass.
