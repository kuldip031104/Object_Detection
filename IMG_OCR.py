from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import re


def basic_clean(text):
    text = text.upper()
    return re.sub(r"[^A-Z0-9]", "", text)

def fix_ocr(text):
    corrected = ""

    for ch in text:
        if ch in ["O", "Q"]:        
            corrected += "0"
        elif ch == "I":             
            corrected += "1"
        elif ch == "Z":             
            corrected += "2"
        elif ch == "S":             
            corrected += "5"
        elif ch == "B":             
            corrected += "8"
        else:
            corrected += ch

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


def preprocess(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


model = YOLO(r"C:\Users\KULDIP\OneDrive\Desktop\Bluepixel\DL\YOLO\Num_plate\runs\detect\train3\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=True)

input_path = r"C:\Users\KULDIP\Downloads\img6.webp"
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
prep = preprocess(plate)
ocr = reader.readtext(prep, detail=0)

raw = "".join(ocr)
final_text = clean_indian_plate(raw)

print("RAW OCR:", raw)
print("CLEANED:", final_text)


cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame, final_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

while True:
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
