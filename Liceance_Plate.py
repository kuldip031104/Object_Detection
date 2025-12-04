from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\KULDIP\OneDrive\Desktop\Bluepixel\DL\YOLO\Num_plate\runs\detect\train\weights\best.pt")

input_path = r"car_num.mp4"
output_path = r"runs\detect\car_num.mp4"

cap = cv2.VideoCapture(input_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:  # <-- break can only be used inside this loop
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Video", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
    
    
out.write(annotated_frame)
    
cap.release()
out.release()
cv2.destroyAllWindows()

print("Done! Saved:", output_path)

