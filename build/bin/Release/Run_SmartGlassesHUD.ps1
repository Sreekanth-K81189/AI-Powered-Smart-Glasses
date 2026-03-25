# Launcher: prepend vcpkg, CUDA, and cuDNN to PATH, then run SmartGlassesHUD.exe
# Run from: project root or build\bin\Release\. Set VCPKG_ROOT, CUDA_PATH, CUDNN_PATH if needed.
$ErrorActionPreference = "Stop"
$vcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { "D:\vcpkg" }
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$vcpkgBin = Join-Path $vcpkgRoot "installed\x64-windows\bin"
$cudaBin = Join-Path $cudaPath "bin"
# cuDNN - ORT GPU needs cudnn64_9.dll (LoadLibrary 126 fix)
$cudnnPaths = @()
if ($env:CUDNN_PATH) { $cudnnPaths += $env:CUDNN_PATH, (Join-Path $env:CUDNN_PATH "bin") }
$cudnnPaths += "C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64",
               "C:\Program Files\NVIDIA\CUDNN\v9.3\bin\12.6\x64",
               "C:\Program Files\NVIDIA\CUDNN\v9.2\bin\12.4\x64"
$cudnnBin = $null
foreach ($p in $cudnnPaths) {
    if ($p -and (Test-Path (Join-Path $p "cudnn64_9.dll"))) { $cudnnBin = $p; break }
}
$pathParts = @($vcpkgBin, $cudaBin)
if ($cudnnBin) { $pathParts += $cudnnBin }
$env:PATH = ($pathParts + $env:PATH) -join ";"
# Exe: same dir as script (build\bin\Release), or build\bin\Release when run from project root
$exe = Join-Path $PSScriptRoot "SmartGlassesHUD.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path $PSScriptRoot "smart_glasses_hud.exe" }
if (-not (Test-Path $exe)) {
    $alt = Join-Path $PSScriptRoot "build\bin\Release\SmartGlassesHUD.exe"
    if (Test-Path $alt) { $exe = $alt }
}
if (-not (Test-Path $exe)) {
    Write-Error "SmartGlassesHUD.exe not found. Build the project first (cmake --build build --config Release)."
    exit 1
}
$runDir = [System.IO.Path]::GetDirectoryName($exe)
Push-Location $runDir
try { & $exe @args; exit $LASTEXITCODE }
finally { Pop-Location }
