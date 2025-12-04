from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# Load YOLO model
model = YOLO(r"C:\Users\KULDIP\OneDrive\Desktop\Bluepixel\DL\YOLO\Num_plate\runs\detect\train3\weights\best.pt")

# Load OCR
reader = easyocr.Reader(['en'], gpu=True)

# Video paths
input_path = r"video.mp4"
output_path = r"runs\detect\video.mp4"

cap = cv2.VideoCapture(input_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

CONF_THRESHOLD = 0.60


save_file = open("plates.txt", "w")
max_conf = 0
best_plate_text = ""   # store best detected text


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    det_boxes = results[0].boxes

    if len(det_boxes) == 0:
        cv2.imshow("YOLO + OCR", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    confidences = det_boxes.conf.cpu().numpy()
    best_index = np.argmax(confidences)
    best_box = det_boxes[best_index]

    best_conf = float(best_box.conf.cpu().numpy())

    if best_conf < CONF_THRESHOLD:
        cv2.imshow("YOLO + OCR", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Extract coordinates
    x1, y1, x2, y2 = map(int, best_box.xyxy.cpu().numpy()[0])
    plate = frame[y1:y2, x1:x2]

    if plate.size != 0:

        # Preprocess for OCR
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # OCR
        ocr_result = reader.readtext(gray, detail=0)
        text = "".join(ocr_result) if ocr_result else ""

     
        if text != "" and best_conf > max_conf:
            max_conf = best_conf
            best_plate_text = text

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{text} ({best_conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("YOLO + OCR", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()


if best_plate_text != "":
    save_file.write(f"Best Plate: {best_plate_text}\n")
    save_file.write(f"Confidence: {max_conf:.2f}\n")
else:
    save_file.write("No plate detected.\n")

save_file.close()
cv2.destroyAllWindows()

print("Done! Saved video:", output_path)
print("Best plate saved in plates.txt")
