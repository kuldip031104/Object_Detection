from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import re

# -------------------------------
# OCR CLEANING FUNCTIONS
# -------------------------------

def basic_clean(text):
    text = text.upper()
    return re.sub(r"[^A-Z0-9]", "", text)

def fix_ocr(text):
    corrected = ""
    for ch in text:
        if ch in ["O", "Q"]:  corrected += "0"
        elif ch == "I":       corrected += "1"
        elif ch == "Z":       corrected += "2"
        elif ch == "S":       corrected += "5"
        elif ch == "B":       corrected += "8"
        else:                 corrected += ch
    return corrected

def clean_indian_plate(text):
    text = basic_clean(text)
    text = fix_ocr(text)
    text = re.sub(r"^(IND|INO|IN0)", "", text)

    parts = re.findall(r"[A-Z0-9]+", text)
    if not parts:
        return "NOT_RECOGNIZED"

    text = max(parts, key=len)

    text_list = list(text)
    for i in range(len(text_list)):
        if i >= 2 and text_list[i] == "L":
            text_list[i] = "4"

    text = "".join(text_list)

    pattern = r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}"
    match = re.search(pattern, text)

    return match.group(0) if match else text


# ------------------------------------------------------
# PREPROCESS + TEXT DEBUG + SHOW IMAGES
# ------------------------------------------------------
def preprocess_show(plate):

    print("\n======== PREPROCESS DEBUG OUTPUT ========\n")

    print("[0] Original Plate Shape:", plate.shape)
    cv2.imshow("0 - Original Plate", plate)

    # GRAYSCALE
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    print("[1] Grayscale Shape:", gray.shape)
    cv2.imshow("1 - Grayscale", gray)

    # RESIZE ×2
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    print("[2] Resized x2 Shape:", resized.shape)
    cv2.imshow("2 - Resized x2", resized)

    # BLUR
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    print("[3] Gaussian Blur: DONE")
    cv2.imshow("3 - Gaussian Blur", blurred)

    # OTSU THRESHOLD
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("[4] OTSU Threshold: DONE")
    cv2.imshow("4 - OTSU Threshold", th)

    print("\n=========================================\n")

    return th


# -------------------------------
# YOLO + OCR MAIN CODE
# -------------------------------

model = YOLO(r"C:\Users\KULDIP\OneDrive\Desktop\Bluepixel\DL\YOLO\Num_plate\runs\detect\train3\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=True)

input_path = r"C:\Users\KULDIP\Downloads\img5.jpg"
frame = cv2.imread(input_path)

if frame is None:
    print("Image not found!")
    exit()

results = model(frame, verbose=False)
boxes = results[0].boxes

if len(boxes) == 0:
    print("NO DETECTION")
    exit()

conf = boxes.conf.cpu().numpy()
idx = np.argmax(conf)
box = boxes[idx]

x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
plate = frame[y1:y2, x1:x2]

# SHOW PREPROCESS STEPS IN TEXT + IMAGE
prep = preprocess_show(plate)

# OCR
ocr = reader.readtext(prep, detail=0)
raw = "".join(ocr)
final_text = clean_indian_plate(raw)

print("RAW OCR:", raw)
print("CLEANED:", final_text)

# DRAW RESULT
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame, final_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# SHOW FINAL
while True:
    cv2.imshow("FINAL RESULT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
