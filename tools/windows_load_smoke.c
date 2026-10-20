/*
 * Loads DLLs from a Windows bundle with LOAD_WITH_ALTERED_SEARCH_PATH and
 * fails unless every Visual C++ and OpenMP runtime module in the process came
 * from the bundle. Build it with /MT so the host itself loads no runtime DLL.
 *
 * Usage: windows_load_smoke <bundle-dir> <dll>...
 */
#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <wchar.h>

static const wchar_t *const kRuntimePrefixes[] = {
    L"msvcp140", L"vcruntime140", L"concrt140", L"vcomp140", L"libomp140",
};

static int is_runtime_module(const wchar_t *name) {
  for (size_t i = 0; i < sizeof(kRuntimePrefixes) / sizeof(kRuntimePrefixes[0]); ++i) {
    if (_wcsnicmp(name, kRuntimePrefixes[i], wcslen(kRuntimePrefixes[i])) == 0) {
      return 1;
    }
  }
  return 0;
}

/* Returns the number of runtime modules loaded, or -1 when one is outside
 * bundle_dir (or when bundle_dir is NULL and any is loaded). */
static int check_runtime_modules(const wchar_t *bundle_dir, int *saw_msvcp, int *saw_vcruntime) {
  HMODULE modules[1024];
  DWORD needed = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed)) {
    fwprintf(stderr, L"EnumProcessModules failed: %lu\n", GetLastError());
    return -1;
  }
  int count = 0;
  int outside = 0;
  for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i) {
    wchar_t path[MAX_PATH];
    if (!GetModuleFileNameW(modules[i], path, MAX_PATH)) {
      continue;
    }
    wchar_t *name = wcsrchr(path, L'\\');
    name = name ? name + 1 : path;
    if (!is_runtime_module(name)) {
      continue;
    }
    ++count;
    if (_wcsicmp(name, L"msvcp140.dll") == 0) {
      *saw_msvcp = 1;
    }
    if (_wcsicmp(name, L"vcruntime140.dll") == 0) {
      *saw_vcruntime = 1;
    }
    size_t dir_length = (size_t)(name - path);
    int in_bundle = bundle_dir != NULL && dir_length > 0 &&
                    wcslen(bundle_dir) == dir_length - 1 &&
                    _wcsnicmp(path, bundle_dir, dir_length - 1) == 0;
    wprintf(L"%ls: %ls\n", in_bundle ? L"bundled" : L"OUTSIDE BUNDLE", path);
    outside |= !in_bundle;
  }
  return outside ? -1 : count;
}

int wmain(int argc, wchar_t **argv) {
  if (argc < 3) {
    fwprintf(stderr, L"Usage: %ls <bundle-dir> <dll>...\n", argv[0]);
    return 2;
  }
  wchar_t bundle_dir[MAX_PATH];
  DWORD length = GetFullPathNameW(argv[1], MAX_PATH, bundle_dir, NULL);
  if (length == 0 || length >= MAX_PATH) {
    fwprintf(stderr, L"Cannot resolve %ls\n", argv[1]);
    return 2;
  }
  if (bundle_dir[length - 1] == L'\\') {
    bundle_dir[length - 1] = L'\0';
  }

  int saw_msvcp = 0;
  int saw_vcruntime = 0;
  if (check_runtime_modules(NULL, &saw_msvcp, &saw_vcruntime) != 0) {
    fwprintf(stderr, L"The host already loaded a runtime DLL; build it with /MT.\n");
    return 1;
  }

  for (int i = 2; i < argc; ++i) {
    wchar_t path[MAX_PATH];
    if (swprintf(path, MAX_PATH, L"%ls\\%ls", bundle_dir, argv[i]) < 0) {
      return 2;
    }
    HMODULE module = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (module == NULL) {
      fwprintf(stderr, L"LoadLibraryExW(%ls) failed: error %lu\n", path, GetLastError());
      return 1;
    }
    wprintf(L"loaded %ls\n", path);
    FARPROC probe = GetProcAddress(module, "llama_dart_tts_api_version");
    if (probe != NULL) {
      wprintf(L"llama_dart_tts_api_version() = %u\n", ((unsigned (*)(void))probe)());
    }
  }

  if (check_runtime_modules(bundle_dir, &saw_msvcp, &saw_vcruntime) < 0) {
    fwprintf(stderr, L"A runtime DLL was loaded from outside %ls.\n", bundle_dir);
    return 1;
  }
  if (!saw_msvcp || !saw_vcruntime) {
    fwprintf(stderr, L"msvcp140.dll and vcruntime140.dll were not both loaded.\n");
    return 1;
  }
  return 0;
}
