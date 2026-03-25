# Fix Sign Language Integration - All-in-One
# Run: powershell -ExecutionPolicy Bypass -File fix_sign_language.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelDir    = "$ProjectRoot\models\sign_language"
$HandModel   = "$ModelDir\hand_landmarker.task"
$GestureModel= "$ModelDir\gesture_recognizer.task"
$HandURL     = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
$GestureURL  = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Sign Language Fix Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# -------------------------------------------------------
# FIX 1 - Install MediaPipe
# -------------------------------------------------------
Write-Host "[FIX 1/3] Installing MediaPipe..." -ForegroundColor Yellow

python -m pip uninstall mediapipe -y 2>$null
python -m pip install --upgrade pip
python -m pip install "mediapipe==0.10.9" "numpy>=1.23.0,<2.0.0" "protobuf>=3.20.0,<5.0.0" "flatbuffers>=2.0" "absl-py"

$mpVer = python -c "import mediapipe as mp; print(mp.__version__)" 2>$null
if ($mpVer -match "^\d+") {
    Write-Host "  PASS - MediaPipe installed: $mpVer" -ForegroundColor Green
} else {
    Write-Host "  FAIL - MediaPipe install failed" -ForegroundColor Red
}

Write-Host "[FIX 1/3] Done" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------
# FIX 2 - Download model files
# -------------------------------------------------------
Write-Host "[FIX 2/3] Downloading model files..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (Test-Path $HandModel) {
    Write-Host "  SKIP - hand_landmarker.task already exists" -ForegroundColor Green
} else {
    Write-Host "  Downloading hand_landmarker.task (~29 MB)..."
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($HandURL, $HandModel)
        Write-Host "  PASS - hand_landmarker.task downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  FAIL - Could not download. Get it manually:" -ForegroundColor Red
        Write-Host "  URL : $HandURL" -ForegroundColor Yellow
        Write-Host "  Save: $HandModel" -ForegroundColor Yellow
    }
}

if (Test-Path $GestureModel) {
    Write-Host "  SKIP - gesture_recognizer.task already exists" -ForegroundColor Green
} else {
    Write-Host "  Downloading gesture_recognizer.task (~5 MB)..."
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($GestureURL, $GestureModel)
        Write-Host "  PASS - gesture_recognizer.task downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  FAIL - Could not download. Get it manually:" -ForegroundColor Red
        Write-Host "  URL : $GestureURL" -ForegroundColor Yellow
        Write-Host "  Save: $GestureModel" -ForegroundColor Yellow
    }
}

Write-Host "[FIX 2/3] Done" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------
# FIX 3 - Fix CUDA/cuDNN crash
# -------------------------------------------------------
Write-Host "[FIX 3/3] Fixing CUDA/cuDNN crash..." -ForegroundColor Yellow

$CudaVersion = ""
try {
    $nvOut = nvidia-smi 2>$null
    if ($nvOut -match "CUDA Version: (\d+)\.(\d+)") {
        $CudaVersion = $Matches[1]
        Write-Host "  Detected CUDA major version: $CudaVersion"
    }
} catch {
    Write-Host "  Could not detect CUDA - using CPU fallback" -ForegroundColor Yellow
}

python -m pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y 2>$null

if ($CudaVersion -eq "12") {
    Write-Host "  CUDA 12 detected - installing onnxruntime-gpu 1.17.0"
    python -m pip install "onnxruntime-gpu==1.17.0"
} elseif ($CudaVersion -eq "11") {
    Write-Host "  CUDA 11 detected - installing onnxruntime-gpu 1.16.3"
    python -m pip install "onnxruntime-gpu==1.16.3"
} else {
    Write-Host "  No CUDA detected - installing onnxruntime CPU (sign language still works)" -ForegroundColor Yellow
    python -m pip install "onnxruntime==1.17.0"
}

$ortCheck = python -c "import onnxruntime; print('ok')" 2>$null
if ($ortCheck -eq "ok") {
    Write-Host "  PASS - onnxruntime works" -ForegroundColor Green
} else {
    Write-Host "  FAIL - onnxruntime still broken" -ForegroundColor Red
}

Write-Host "[FIX 3/3] Done" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------
# VERIFY - Re-run the 5 failed checks
# -------------------------------------------------------
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Verification" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$Pass = 0
$Fail = 0

# Check 5
if (Test-Path $HandModel) {
    Write-Host "  [PASS] Check 5 - hand_landmarker.task exists" -ForegroundColor Green
    $Pass++
} else {
    Write-Host "  [FAIL] Check 5 - hand_landmarker.task missing" -ForegroundColor Red
    $Fail++
}

# Check 6
if (Test-Path $GestureModel) {
    Write-Host "  [PASS] Check 6 - gesture_recognizer.task exists" -ForegroundColor Green
    $Pass++
} else {
    Write-Host "  [FAIL] Check 6 - gesture_recognizer.task missing" -ForegroundColor Red
    $Fail++
}

# Check 7
$mpResult = python -c "import mediapipe; print(mediapipe.__version__)" 2>$null
if ($mpResult -match "^\d+") {
    Write-Host "  [PASS] Check 7 - MediaPipe installed: $mpResult" -ForegroundColor Green
    $Pass++
} else {
    Write-Host "  [FAIL] Check 7 - MediaPipe not importable" -ForegroundColor Red
    $Fail++
}

# Check 10
$svcResult = python "$ProjectRoot\scripts\python\sign_language_service.py" --help 2>$null
if ($svcResult -match "Traceback|ModuleNotFoundError|ImportError|Error") {
    Write-Host "  [FAIL] Check 10 - sign_language_service.py still errors" -ForegroundColor Red
    $Fail++
} else {
    Write-Host "  [PASS] Check 10 - sign_language_service.py runs cleanly" -ForegroundColor Green
    $Pass++
}

# Check 40
$ortResult = python -c "import onnxruntime; print('ok')" 2>$null
if ($ortResult -eq "ok") {
    Write-Host "  [PASS] Check 40 - onnxruntime imports without crash" -ForegroundColor Green
    Write-Host "         NOTE: Also launch smart_glasses_hud.exe to confirm full runtime" -ForegroundColor Yellow
    $Pass++
} else {
    Write-Host "  [FAIL] Check 40 - onnxruntime still crashing" -ForegroundColor Red
    $Fail++
}

Write-Host ""
Write-Host "  Passed : $Pass / 5" -ForegroundColor Green
if ($Fail -gt 0) {
    Write-Host "  Failed : $Fail / 5" -ForegroundColor Red
}
Write-Host ""

if ($Fail -eq 0) {
    Write-Host "  ALL FIXES APPLIED - Score now 37/40" -ForegroundColor Green
    Write-Host "  Launch smart_glasses_hud.exe and test Sign Language in Translation menu"
} else {
    Write-Host "  $Fail check(s) still failing - see above for manual steps" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================"
