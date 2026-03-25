# ================================================================
#   AI-Powered Smart Glasses — One-Click Setup Script
#   Run in PowerShell as Administrator:
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\setup.ps1
# ================================================================

$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"   # speeds up Invoke-WebRequest

$PROJECT_DIR = $PSScriptRoot
Set-Location $PROJECT_DIR

# ── Colour helpers ───────────────────────────────────────────────
function OK  ($m) { Write-Host "  [OK] $m"      -ForegroundColor Green  }
function INF ($m) { Write-Host "  [..] $m"      -ForegroundColor Cyan   }
function WRN ($m) { Write-Host "  [!!] $m"      -ForegroundColor Yellow }
function ERR ($m) { Write-Host "  [XX] $m"      -ForegroundColor Red    }
function HDR ($m) {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
    Write-Host "  $m" -ForegroundColor White
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
}

# ── Download helper (with progress + retry) ──────────────────────
function Download ($url, $dest, $label) {
    if (Test-Path $dest) { OK "$label already exists — skipping"; return }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    INF "Downloading $label..."
    $attempt = 0
    while ($attempt -lt 3) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            OK "$label downloaded"
            return
        } catch {
            $attempt++
            WRN "Attempt $attempt failed — retrying..."
            Start-Sleep 3
        }
    }
    ERR "Failed to download $label after 3 attempts."
    ERR "Manual URL: $url"
}

# ================================================================
HDR "STEP 1 — Checking Prerequisites"
# ================================================================

# ── winget ───────────────────────────────────────────────────────
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    ERR "winget not found. Install App Installer from Microsoft Store, then re-run."
    exit 1
}
OK "winget found"

# ── Git ──────────────────────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    INF "Installing Git..."
    winget install -e --id Git.Git --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}
OK "Git: $(git --version)"

# ── Python 3.10 ──────────────────────────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    INF "Installing Python 3.10..."
    winget install -e --id Python.Python.3.10 --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}
OK "Python: $(python --version)"

# ── CMake ─────────────────────────────────────────────────────────
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    INF "Installing CMake..."
    winget install -e --id Kitware.CMake --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}
OK "CMake: $(cmake --version | Select-Object -First 1)"

# ── Visual Studio Build Tools ────────────────────────────────────
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    INF "Installing Visual Studio 2022 Build Tools (C++ workload)..."
    winget install -e --id Microsoft.VisualStudio.2022.BuildTools --silent `
        --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    WRN "VS Build Tools installed. You may need to restart and re-run this script."
}
OK "Visual Studio Build Tools found"

# ── Java (for BFG / optional tooling) ────────────────────────────
if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    INF "Installing OpenJDK 21..."
    winget install -e --id Microsoft.OpenJDK.21 --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}
OK "Java: $(java -version 2>&1 | Select-Object -First 1)"

# ── Tesseract OCR ─────────────────────────────────────────────────
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    INF "Installing Tesseract OCR..."
    winget install -e --id UB-Mannheim.TesseractOCR --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}
OK "Tesseract found"

# ================================================================
HDR "STEP 2 — CUDA 12.x Check"
# ================================================================

$cudaOK = $false
try {
    $nvcc = & nvcc --version 2>&1
    if ($nvcc -match "release 12") { $cudaOK = $true; OK "CUDA 12.x already installed" }
    else { WRN "CUDA found but not version 12: $nvcc" }
} catch { }

if (-not $cudaOK) {
    WRN "CUDA 12 not detected."
    Write-Host ""
    Write-Host "  Please install CUDA 12.x manually:" -ForegroundColor Yellow
    Write-Host "  https://developer.nvidia.com/cuda-12-0-0-download-archive" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Also install cuDNN 9.x for CUDA 12:" -ForegroundColor Yellow
    Write-Host "  https://developer.nvidia.com/cudnn-downloads" -ForegroundColor Cyan
    Write-Host ""
    $cont = Read-Host "  Have you already installed CUDA 12? (y/n)"
    if ($cont -ne "y") {
        WRN "Please install CUDA 12, then re-run this script."
        exit 0
    }
}

# ================================================================
HDR "STEP 3 — Python Packages"
# ================================================================

INF "Upgrading pip..."
python -m pip install --upgrade pip --quiet

$packages = @(
    "ultralytics",          # YOLOv8 / YOLO11
    "opencv-python",        # OpenCV
    "mediapipe",            # Sign language / hand landmarks
    "openai-whisper",       # Speech to text
    "pytesseract",          # OCR Python binding
    "numpy",
    "requests",
    "gdown",                # Google Drive downloader
    "torch",                # PyTorch (CPU — CUDA version installed below)
    "torchvision",
    "torchaudio",
    "flask",                # Python microservices bridge
    "onnxruntime-gpu"       # ONNX Runtime with CUDA
)

INF "Installing Python packages (this may take a few minutes)..."
python -m pip install $packages --quiet
OK "Python packages installed"

# ── PyTorch with CUDA 12 ─────────────────────────────────────────
INF "Installing PyTorch with CUDA 12.1 support..."
python -m pip install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu121 --quiet
OK "PyTorch CUDA installed"

# ================================================================
HDR "STEP 4 — Downloading ONNX Runtime (CUDA build)"
# ================================================================

$onnxDir  = "$PROJECT_DIR\third_party\onnxruntime-cuda"
$onnxZip  = "$env:TEMP\onnxruntime-cuda.zip"
$onnxUrl  = "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-gpu-1.20.1.zip"

if (-not (Test-Path "$onnxDir\lib\onnxruntime.dll")) {
    Download $onnxUrl $onnxZip "ONNX Runtime CUDA v1.20.1"
    INF "Extracting ONNX Runtime..."
    Expand-Archive -Path $onnxZip -DestinationPath $env:TEMP -Force
    $extracted = Get-ChildItem "$env:TEMP\onnxruntime-win-x64-gpu*" -Directory | Select-Object -First 1
    if ($extracted) {
        New-Item -ItemType Directory -Force -Path $onnxDir | Out-Null
        Copy-Item "$($extracted.FullName)\*" $onnxDir -Recurse -Force
        OK "ONNX Runtime extracted to third_party/onnxruntime-cuda"
    }
} else {
    OK "ONNX Runtime already present"
}

# ================================================================
HDR "STEP 5 — Downloading Model Files"
# ================================================================

function DLModel ($url, $dest, $name) { Download $url "$PROJECT_DIR\$dest" $name }

# ── YOLO models (official Ultralytics releases) ──────────────────
DLModel "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" `
        "models\weights\yolov8n.pt"  "YOLOv8n (nano)"
DLModel "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt" `
        "models\weights\yolov8x.pt"  "YOLOv8x (large)"
DLModel "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" `
        "models\weights\yolo11n.pt"  "YOLO11n (nano)"
DLModel "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt" `
        "models\weights\yolo11x.pt"  "YOLO11x (large)"

# ── Export YOLOv8x to ONNX FP16 (needed for real-time inference) ─
$onnxModel = "$PROJECT_DIR\models\yolo\yolov8x_fp16.onnx"
if (-not (Test-Path $onnxModel)) {
    INF "Exporting YOLOv8x to ONNX FP16 (takes ~2 minutes on first run)..."
    New-Item -ItemType Directory -Force -Path "$PROJECT_DIR\models\yolo" | Out-Null
    python -c @"
from ultralytics import YOLO
import shutil, os
m = YOLO('$($PROJECT_DIR.Replace('\','/'))/models/weights/yolov8x.pt')
m.export(format='onnx', half=True, imgsz=640)
src = '$($PROJECT_DIR.Replace('\','/'))/models/weights/yolov8x.onnx'
dst = '$($onnxModel.Replace('\','/'))'
if os.path.exists(src): shutil.move(src, dst)
print('Export done')
"@
    if (Test-Path $onnxModel) { OK "YOLOv8x ONNX FP16 exported" }
    else { WRN "ONNX export failed — you can do it manually later" }
} else {
    OK "YOLOv8x ONNX FP16 already exists"
}

# ── MediaPipe models (official Google storage) ───────────────────
DLModel "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
        "models\sign_language\hand_landmarker.task"    "MediaPipe Hand Landmarker"
DLModel "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" `
        "models\sign_language\gesture_recognizer.task" "MediaPipe Gesture Recognizer"

# ── Face detection models ────────────────────────────────────────
DLModel "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
        "models\face\face_detection_yunet.onnx"        "YuNet Face Detection"
DLModel "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" `
        "models\face\haarcascade_frontalface_default.xml" "Haar Cascade Face"

# ── OCR models ───────────────────────────────────────────────────
DLModel "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb" `
        "models\ocr\frozen_east_text_detection.pb"     "EAST Text Detector"

# Copy Tesseract eng.traineddata to models/tessdata
$tessdata = "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata"
if (Test-Path $tessdata) {
    New-Item -ItemType Directory -Force -Path "$PROJECT_DIR\models\tessdata" | Out-Null
    Copy-Item $tessdata "$PROJECT_DIR\models\tessdata\eng.traineddata" -Force
    OK "Tesseract eng.traineddata copied"
} else {
    WRN "Tesseract traineddata not found at default path — copy manually if needed"
}

# ── Whisper model (download via Python) ──────────────────────────
INF "Pre-downloading Whisper base.en model..."
python -c "import whisper; whisper.load_model('base.en'); print('Whisper base.en ready')"
OK "Whisper model ready"

# ================================================================
HDR "STEP 6 — Git LFS Pull (remaining tracked files)"
# ================================================================

if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
    INF "Pulling Git LFS objects..."
    git lfs pull
    OK "Git LFS pull complete"
} else {
    INF "Installing Git LFS..."
    winget install -e --id GitHub.GitLFS --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
    git lfs install
    git lfs pull
    OK "Git LFS pull complete"
}

# ================================================================
HDR "STEP 7 — Build Project"
# ================================================================

$buildDir = "$PROJECT_DIR\build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

INF "Running CMake configure..."
cmake -S $PROJECT_DIR -B $buildDir -DCMAKE_BUILD_TYPE=Release 2>&1 | Tee-Object -Variable cmakeOut
if ($LASTEXITCODE -ne 0) {
    ERR "CMake configuration failed. Check output above."
    WRN "You may need to open the project in Visual Studio manually."
} else {
    INF "Building project (Release)..."
    cmake --build $buildDir --config Release --parallel
    if ($LASTEXITCODE -eq 0) { OK "Build succeeded!" }
    else { ERR "Build failed — check errors above." }
}

# ================================================================
HDR "SETUP COMPLETE"
# ================================================================

Write-Host ""
Write-Host "  Summary of what was installed:" -ForegroundColor White
Write-Host "  ✅  Python packages (ultralytics, mediapipe, whisper, opencv...)" -ForegroundColor Green
Write-Host "  ✅  PyTorch with CUDA 12.1"                                        -ForegroundColor Green
Write-Host "  ✅  ONNX Runtime CUDA v1.20.1"                                     -ForegroundColor Green
Write-Host "  ✅  YOLO models (v8n, v8x, 11n, 11x + ONNX FP16)"                 -ForegroundColor Green
Write-Host "  ✅  MediaPipe models (hand landmarker + gesture recognizer)"        -ForegroundColor Green
Write-Host "  ✅  Face detection models (YuNet + Haar Cascade)"                   -ForegroundColor Green
Write-Host "  ✅  OCR models (EAST + Tesseract eng)"                              -ForegroundColor Green
Write-Host "  ✅  Whisper base.en model"                                          -ForegroundColor Green
Write-Host ""
Write-Host "  To run the project:" -ForegroundColor White
Write-Host "  .\build\bin\Release\SmartGlasses.exe" -ForegroundColor Cyan
Write-Host ""
WRN "NOTE: CUDA 12 + cuDNN 9 must be installed manually if not already:"
Write-Host "  CUDA:   https://developer.nvidia.com/cuda-12-0-0-download-archive" -ForegroundColor Cyan
Write-Host "  cuDNN:  https://developer.nvidia.com/cudnn-downloads" -ForegroundColor Cyan
Write-Host ""
