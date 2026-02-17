# Platform Backend Strategy

## Default Strategy

| Platform | Default Backend |
|---|---|
| Android (arm64/x64) | OpenCL |
| iOS | Metal |
| macOS | Metal |
| Linux x64/arm64 | Vulkan |
| Windows x64 | Vulkan |

## Fallback Strategy (runtime policy target)

| Platform | Fallback Chain |
|---|---|
| Android | OpenCL -> Vulkan -> CPU |
| iOS/macOS | Metal -> CPU |
| Linux | Vulkan -> CPU |
| Windows | Vulkan -> CPU |

## Configurable Backends

| Platform | Configurable options |
|---|---|
| Android | `opencl`, `vulkan`, `cpu` |
| Apple | `metal`, `cpu` |
| Linux | `vulkan`, `cpu`, `cuda`, `zendnn` |
| Windows | `vulkan`, `cpu`, `cuda` |

## CPU Optimization Notes

- x86: AVX2/AVX512/FMA/F16C options are available upstream.
- ARM64: dotprod/i8mm/SVE/SME paths exist upstream.
- AMD server CPUs: ZenDNN can accelerate matrix multiplication workloads.

## Current Constraints

- CUDA builds require CUDA toolchain availability (`nvcc`) on the runner.
- Android OpenCL requires OpenCL headers/ICD loader in the NDK/sysroot.
- ZenDNN currently targets Linux x64.

