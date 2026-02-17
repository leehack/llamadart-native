# build_windows.ps1 <backend> [clean]
# backends: vulkan | cpu | cuda

param (
    [string]$Backend = "vulkan",
    [string]$Clean = "",
    [string]$VulkanSdk = ""
)

$BuildDir = "build-$Backend"
if ($Clean -eq "clean" -and (Test-Path $BuildDir)) {
    Remove-Item -Path $BuildDir -Recurse -Force
}

$CmakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF"
)

switch ($Backend) {
    "vulkan" {
        if ($VulkanSdk -ne "") {
            $env:VULKAN_SDK = $VulkanSdk
        }
        if (-not $env:VULKAN_SDK) {
            $Glslc = Get-Command glslc.exe -ErrorAction SilentlyContinue
            if ($Glslc) {
                $SdkBin = [System.IO.Path]::GetDirectoryName($Glslc.Source)
                $SdkRoot = [System.IO.Path]::GetDirectoryName($SdkBin)
                if ((Test-Path "$SdkRoot/Include/vulkan/vulkan.h") -or (Test-Path "$SdkRoot/Lib/vulkan-1.lib")) {
                    $env:VULKAN_SDK = $SdkRoot
                }
            }
        }

        if (-not $env:VULKAN_SDK) {
            Write-Error "Vulkan backend requested but VULKAN_SDK not found."
            exit 1
        }

        $SdkRoot = $env:VULKAN_SDK.Replace('\\', '/')
        $CmakeArgs += "-DVulkan_ROOT=$SdkRoot"
        $CmakeArgs += "-DVulkan_INCLUDE_DIR=$SdkRoot/Include"

        $VulkanLib = Get-ChildItem -Path "$SdkRoot/Lib" -Filter "vulkan-1.lib" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $VulkanLib) {
            $VulkanLib = Get-ChildItem -Path $SdkRoot -Filter "vulkan-1.lib" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        }
        if ($VulkanLib) {
            $CmakeArgs += "-DVulkan_LIBRARY=$($VulkanLib.FullName.Replace('\\', '/'))"
        }

        $GlslcExe = Get-ChildItem -Path $SdkRoot -Filter "glslc.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($GlslcExe) {
            $CmakeArgs += "-DVulkan_GLSLC_EXECUTABLE=$($GlslcExe.FullName.Replace('\\', '/'))"
        }

        $CmakeArgs += "-DGGML_VULKAN=ON"
        $CmakeArgs += "-DGGML_CUDA=OFF"
    }
    "cuda" {
        if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
            Write-Error "CUDA backend requested but nvcc.exe is not available in PATH."
            exit 1
        }
        $CmakeArgs += "-DGGML_CUDA=ON"
        $CmakeArgs += "-DGGML_VULKAN=OFF"
    }
    "cpu" {
        $CmakeArgs += "-DGGML_CUDA=OFF"
        $CmakeArgs += "-DGGML_VULKAN=OFF"
    }
    default {
        Write-Error "Invalid backend '$Backend'. Use vulkan|cpu|cuda."
        exit 1
    }
}

if (-not (Test-Path $BuildDir)) {
    New-Item -Path $BuildDir -ItemType Directory | Out-Null
}

function Get-CMake {
    $cmd = Get-Command "cmake" -ErrorAction SilentlyContinue
    if ($cmd) { return "cmake" }

    $paths = @(
        "$env:USERPROFILE\scoop\apps\cmake\current\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Get-CompilerCache {
    $scc = Get-Command "sccache" -ErrorAction SilentlyContinue
    if ($scc) { return "sccache" }
    $ccc = Get-Command "ccache" -ErrorAction SilentlyContinue
    if ($ccc) { return "ccache" }
    return $null
}

$cache = Get-CompilerCache
if ($cache) {
    $CmakeArgs += "-DCMAKE_C_COMPILER_LAUNCHER=$cache"
    $CmakeArgs += "-DCMAKE_CXX_COMPILER_LAUNCHER=$cache"
}

$CmakeExe = Get-CMake
if (-not $CmakeExe) {
    Write-Error "CMake not found."
    exit 1
}

& "$CmakeExe" -G Ninja -S . -B $BuildDir @CmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& "$CmakeExe" --build $BuildDir -j 8
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$LibDir = "bin/windows/x64"
if (Test-Path $LibDir) {
    Remove-Item -Path $LibDir -Recurse -Force
}
New-Item -Path $LibDir -ItemType Directory -Force | Out-Null

Get-ChildItem -Path $BuildDir -Filter *.dll -Recurse | ForEach-Object {
    $Name = $_.Name
    $DestName = if ($Name -eq "llamadart.dll") { "libllamadart.dll" } else { $Name }
    $DestPath = Join-Path $LibDir $DestName
    if (-not (Test-Path $DestPath)) {
        Copy-Item -Path $_.FullName -Destination $DestPath -Force
    }
}

Write-Host "Windows build complete backend=$Backend: $LibDir\\libllamadart.dll"
