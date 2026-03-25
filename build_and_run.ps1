# Build and run Smart Glasses HUD in one command
# Usage (from project root):  powershell -ExecutionPolicy Bypass -File .\build_and_run.ps1

$ErrorActionPreference = "Stop"
$projRoot = $PSScriptRoot

function Import-DotEnv([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    Get-Content -LiteralPath $path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { return }
        $k = $line.Substring(0, $eq).Trim()
        $v = $line.Substring($eq + 1).Trim()
        if ($k) { Set-Item -Path "Env:$k" -Value $v }
    }
}

function Normalize-Path([string]$p) {
    if (-not $p) { return $null }
    try { return (Resolve-Path -LiteralPath $p -ErrorAction Stop).Path }
    catch { return $p }
}

function Stop-RunningApp() {
    # If the app is still running (or a previous run didn't exit cleanly), it can lock build outputs.
    foreach ($name in @("smart_glasses_hud", "SmartGlassesHUD")) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                Write-Host "Stopping running process: $($_.Name) (pid=$($_.Id))" -ForegroundColor Yellow
                Stop-Process -Id $_.Id -Force -ErrorAction Stop
            } catch {}
        }
    }
}

function Remove-DirectoryWithRetry([string]$dir) {
    if (-not (Test-Path -LiteralPath $dir)) { return }
    Stop-RunningApp
    $max = 8
    for ($i = 1; $i -le $max; $i++) {
        try {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop
            return
        } catch {
            if ($i -eq $max) { throw }
            Start-Sleep -Milliseconds (250 * $i)
        }
    }
}

Write-Host "=== Configuring (Release) ===" -ForegroundColor Cyan
Set-Location $projRoot

# Load local environment (VCPKG_ROOT, etc.) so find_package() works.
Import-DotEnv (Join-Path $projRoot ".env")

# Prefer explicit vcpkg toolchain to avoid relying on CMake cache state.
$toolchain = $null
if ($env:VCPKG_ROOT) {
    $tc = Join-Path $env:VCPKG_ROOT "scripts\\buildsystems\\vcpkg.cmake"
    if (Test-Path $tc) { $toolchain = $tc }
}

$cacheTool = $null
foreach ($tool in @("sccache", "ccache")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { $cacheTool = $cmd.Source; break }
}
# WinGet portable installs often create shims in %LOCALAPPDATA%\Microsoft\WinGet\Links
if (-not $cacheTool -and $env:LOCALAPPDATA) {
    $wgSccache = Join-Path $env:LOCALAPPDATA "Microsoft\\WinGet\\Links\\sccache.exe"
    if (Test-Path -LiteralPath $wgSccache) { $cacheTool = $wgSccache }
}
if ($cacheTool) {
    Write-Host "Compiler cache enabled: $cacheTool" -ForegroundColor Green
} else {
    Write-Host "Compiler cache not found (install sccache for faster rebuilds)." -ForegroundColor DarkGray
}

$buildDir = Join-Path $projRoot "build"
$cacheFile = Join-Path $buildDir "CMakeCache.txt"
if ($toolchain -and (Test-Path -LiteralPath $cacheFile)) {
    $cachedTc = (Select-String -LiteralPath $cacheFile -Pattern '^CMAKE_TOOLCHAIN_FILE:FILEPATH=' -ErrorAction SilentlyContinue | Select-Object -First 1).Line
    if ($cachedTc) {
        $cachedTcPath = $cachedTc.Split('=', 2)[1]
        $cachedNorm = Normalize-Path $cachedTcPath
        $toolchainNorm = Normalize-Path $toolchain
        if ($cachedNorm -ne $toolchainNorm) {
            Write-Host "CMake cache has different toolchain; cleaning build/ ..." -ForegroundColor Yellow
            Remove-DirectoryWithRetry $buildDir
        }
    } else {
        # Toolchain not present in cache -> rebuild from scratch to ensure vcpkg packages resolve.
        Write-Host "CMake cache missing toolchain; cleaning build/ ..." -ForegroundColor Yellow
        Remove-DirectoryWithRetry $buildDir
    }
}

$cmakeArgs = @(
    "-S", ".", "-B", "build",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CCACHE=OFF",
    "-Wno-dev",
    "-Wno-deprecated"
)
# Only pass toolchain on first configure (or after a clean). Passing it on an
# already-configured build dir can trigger "Manually-specified variables were not used".
if ($toolchain -and -not (Test-Path -LiteralPath $cacheFile)) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$toolchain"
}

# Use sccache/ccache as compiler launcher if available.
if ($cacheTool) {
    $cmakeArgs += "-DCMAKE_C_COMPILER_LAUNCHER=$cacheTool"
    $cmakeArgs += "-DCMAKE_CXX_COMPILER_LAUNCHER=$cacheTool"
}

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Building (Release) ===" -ForegroundColor Cyan
# Keep build logs readable: minimize MSBuild output (warnings still count, but won't spam template context).
cmake --build build --config Release --target smart_glasses_hud -- /verbosity:minimal "/clp:ErrorsOnly;Summary"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Launching ===" -ForegroundColor Green

# Prefer the launcher script that CMake copies next to the built exe.
$outDir = Join-Path $projRoot "build\bin\Release"
$runScript = Join-Path $outDir "Run_SmartGlassesHUD.ps1"

if (Test-Path $runScript) {
    & $runScript @args
    exit $LASTEXITCODE
}

# Fallback: run via root launcher (it can locate the exe too).
$rootRun = Join-Path $projRoot "Run_SmartGlassesHUD.ps1"
if (Test-Path $rootRun) {
    & $rootRun @args
    exit $LASTEXITCODE
}

throw "Run script not found. Expected either '$runScript' or '$rootRun'."

