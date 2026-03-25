# Build setup (after vcpkg partial install)

## What happened
- **CMake, Git, VS Build Tools** – installed OK  
- **vcpkg** – installed OK  
- **espeak-ng** – not in vcpkg (skipped)  
- **OpenCV / Tesseract** – vcpkg failed on dependency `pkgconf`  
- **GLFW** – installed OK via vcpkg  

## Option A – Use prebuilt OpenCV (recommended)

1. **Download OpenCV for Windows**  
   https://github.com/opencv/opencv/releases  
   Get **opencv-4.10.0-windows.exe** (or latest `opencv-*-windows.exe`), run it and extract to **`C:\opencv`** (so you have `C:\opencv\build`).

2. **Build the project** (in PowerShell, from `D:\Final Year Project`):
   ```powershell
   $env:OpenCV_DIR = "C:\opencv\build"
   .\build_with_vcpkg.ps1
   ```
   If you extracted elsewhere, set `OpenCV_DIR` to the folder that **contains** `OpenCVConfig.cmake` (often `...\opencv\build`).

3. **Run the app**
   ```powershell
   .\build\bin\SmartGlassesHUD.exe
   ```

- **TTS:** Disabled (espeak-ng not installed).  
- **OCR:** Disabled unless you install Tesseract separately and point CMake to it.

## Option B – Fix vcpkg and install OpenCV via vcpkg

In **Administrator** PowerShell:

```powershell
cd D:\tools\vcpkg
git pull
.\vcpkg.exe remove pkgconf:x64-windows
.\vcpkg.exe install opencv4[tiff,ffmpeg]:x64-windows
.\vcpkg.exe integrate install
```

Then from `D:\Final Year Project`:

```powershell
.\build_with_vcpkg.ps1
```

(If `opencv4` also fails, use Option A.)
