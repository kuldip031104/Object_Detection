import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# -----------------------------
# LOAD YOLO MODEL + OCR
# -----------------------------
model = YOLO(r"C:\Users\KULDIP\OneDrive\Desktop\Bluepixel\DL\YOLO\Num_plate\runs\detect\train\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# -----------------------------
# REGEX PATTERN FOR INDIAN PLATES
# -----------------------------
plate_pattern = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}")

# -----------------------------
# INDIAN NUMBER PLATE CORRECTION
# -----------------------------
def correct_indian_plate(ocr_text):
    ocr_text = ocr_text.replace(" ", "").upper()

    mapping_num_to_alpha = {'0':'O', '1':'I', '2':'Z', '5':'S', '8':'B'}
    mapping_alpha_to_num = {'O':'0', 'I':'1', 'Z':'2', 'S':'5', 'B':'8'}

    corrected = []

    if len(ocr_text) < 8 or len(ocr_text) > 10:
        return ""

    for i, ch in enumerate(ocr_text):

        # LETTER POSITIONS
        if i in [0, 1, 4, 5, 6]:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""

        # NUMBER POSITIONS
        else:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    return "".join(corrected)


# -----------------------------
# IMAGE PREPROCESSING FOR OCR
# -----------------------------
def preprocess_for_ocr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return []

    # 1. Gray
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # 3. Otsu binary
    _, thresh = cv2.threshold(clahe_img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Inverted binary
    thresh_inv = cv2.bitwise_not(thresh)

    # 5. Sharpening
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    sharp = cv2.filter2D(clahe_img, -1, kernel)

    # Resize helper
    def resize2x(img):
        return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Return all enhanced versions
    return [
        resize2x(gray),
        resize2x(clahe_img),
        resize2x(thresh),
        resize2x(thresh_inv),
        resize2x(sharp)
    ]


# -----------------------------
# BEST EASYOCR RECOGNITION FUNCTION
# -----------------------------
def recognize_plate(plate_crop):
    variants = preprocess_for_ocr(plate_crop)
    if not variants:
        return ""

    candidates = []  # (plate, confidence)

    for img in variants:
        try:
            ocr_results = reader.readtext(
                img,
                detail=1,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
        except:
            continue

        for (_, text, conf) in ocr_results:
            if not text:
                continue

            corrected = correct_indian_plate(text)
            if not corrected:
                continue

            # Must match Indian pattern
            if plate_pattern.match(corrected):
                candidates.append((corrected, conf))

    if not candidates:
        return ""

    # Choose highest confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_plate = candidates[0][0]

    return best_plate


# -----------------------------
# STABILIZATION ACROSS FRAMES
# -----------------------------
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}

def get_box_id(x1, y1, x2, y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")


# -----------------------------
# VIDEO PROCESSING
# -----------------------------
input_video = "video.mp4"
output_video = "output_licence.mp4"

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_video,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(3)), int(cap.get(4)))
)

CONF_THRESH = 0.3

print("🚀 Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        conf = float(box.conf)
        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop license plate area
        plate_crop = frame[y1:y2, x1:x2]

        # OCR
        plate_text = recognize_plate(plate_crop)
        box_id = get_box_id(x1, y1, x2, y2)
        stable_plate = get_stable_plate(box_id, plate_text)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text above plate
        if stable_plate:
            cv2.putText(frame, stable_plate, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output saved:", output_video)
