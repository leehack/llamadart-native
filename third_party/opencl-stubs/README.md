# Android OpenCL Stub Inputs

Optional local fallback paths for Android OpenCL linking:

- `third_party/opencl-stubs/include/CL/cl.h`
- `third_party/opencl-stubs/android/arm64-v8a/libOpenCL.so`
- `third_party/opencl-stubs/android/x86_64/libOpenCL.so`

`tools/build.py android` resolves OpenCL in this order:

1. Environment overrides (`OPENCL_INCLUDE_DIR`, `OPENCL_LIBRARY_ANDROID_<ABI>`)
2. NDK-provided OpenCL stub (if available)
3. This `opencl-stubs/` folder
4. Auto-build from `third_party/OpenCL-ICD-Loader` + `third_party/OpenCL-Headers`

The folders are intentionally empty in git. Provide your own headers/stubs if you do not use the ICD loader auto-build path.
