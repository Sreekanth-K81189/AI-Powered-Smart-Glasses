# Smart Glasses HUD (C++)

AI-powered Smart Glasses assistive technology — C++ build with zero-lag model pre-loading, camera fallback (ESP32 → USB), and Dear ImGui HUD.

## Directory structure

```
./
├── CMakeLists.txt
├── include/          # All .hpp headers
├── src/              # All .cpp sources
├── models/           # Weights: yolov8x_fp16.onnx, tessdata/, cascades
├── deps/             # vcpkg + downloaded runtime deps (optional)
├── build/            # CMake output (bin/, lib/) (generated)
└── README.md
```

## Build

From project root (`./`):

```bash
cmake -B build
cmake --build build
```

Executable: `build/bin/SmartGlassesHUD` (or `SmartGlassesHUD.exe` on Windows).  
Run from `./` so `models/` is found (or run from `build/bin/`; the app will look for `../../models`).

**Models:** Place under `models/`:

- `models/yolo/yolov8x_fp16.onnx` — YOLOv8 for obstacle detection
- `tessdata/` — Tesseract language data (e.g. `eng.traineddata`), or set `TESSDATA_PREFIX`
- `haarcascade_frontalface_default.xml` (or `models/face/`) — OpenCV face cascade

**Optional:** `-DESPEAK_NG_ROOT=/path/to/espeak-ng` for TTS.

## Camera behaviour

1. **Probe** `http://10.112.139.57:81/stream` with a **1.5 s** non-blocking timeout.
2. **Fallback:** if ESP32 fails, try `/dev/video0` … `/dev/video2` (Linux) or indices 0–2 (Windows).
3. **Status:** Prints and HUD show `SOURCE: ESP32`, `SOURCE: USB`, or `SOURCE: NONE`.

## HUD modes

- **Navigation** — YOLO + MoveSafe decision → TTS (e.g. “Path clear”, “Obstacle ahead - slow down”).
- **OCR** — Tesseract; confidence shown; TTS only if confidence > 60%.
- **Face** — OpenCV cascade; “Person detected” → TTS.
- **Sign** — Placeholder; extend with gesture model → text → TTS.
- **Speech** — Placeholder for STT.

All modes use the same **HUDLayer** (unified styling, borders, confidence, detection boxes on the video texture).
