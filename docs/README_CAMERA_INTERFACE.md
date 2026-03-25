# Camera Interface & Task Selection System

## Overview
This application provides a user-friendly interface to:
1. Capture video input from an external camera
2. Display the camera feed in real-time
3. Select and run tasks from your Final Year Project

## Installation

### Prerequisites
- Python 3.7 or higher
- External camera connected to your computer

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python Pillow numpy torch ultralytics easyocr
```

## Usage

### Running the Main Interface
```bash
python camera_interface.py
```

### Features

1. **Camera Feed**
   - Click "Start Camera" to begin capturing from external camera
   - The system will automatically try external camera first (index 1), then fallback to default camera (index 0)
   - Click "Stop Camera" to stop the feed

2. **Task Selection**
   - View all available tasks in the right panel
   - Click on a task to select it
   - Click "Run Selected Task" to execute the chosen task

### Available Tasks
- Facial Detection
- Object Detection
- OCR (Text Recognition)
- Sign Language to Text
- Skin Cancer Detection
- Speech to Text
- Text to Speech
- Chat Bot
- Voice Activated HUD
- Navigation

## Troubleshooting

### Camera Not Detected
- Ensure your external camera is properly connected
- Check if the camera is being used by another application
- Try changing the camera index in the code (0, 1, 2, etc.)

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.7+)

### Task Files Not Found
- Ensure all task Python files exist in the project folder
- Check that task files have proper implementations

## File Structure
```
Final Year Project/
├── camera_interface.py      # Main GUI (camera + task selection)
├── requirements.txt         # Python dependencies
├── Facial_detection.py      # Task modules
├── Object_detection.py
├── OCR.py
├── ... (other task files)
```
