# Install deps: pip install -r requirements.txt (project root)
import sys, os

def run_ocr(image_path):
    try:
        import pytesseract
        from PIL import Image
        import cv2
        import numpy as np

        # Try common Tesseract install paths on Windows
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\chekk\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

        img = cv2.imread(image_path)
        if img is None:
            print("OCR_ERROR: Cannot read image", file=sys.stderr)
            sys.exit(1)

        # Preprocessing for better accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pil_img = Image.fromarray(thresh)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(pil_img, config=config).strip()
        text = " ".join(text.split())  # collapse whitespace
        print(text if text else "OCR_EMPTY")
    except Exception as e:
        print(f"OCR_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ocr_service.py <image_path>", file=sys.stderr)
        sys.exit(1)
    run_ocr(sys.argv[1])
