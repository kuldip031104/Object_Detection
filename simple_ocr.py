import cv2
from easyocr import Reader

# 1. Load Image
image_path = r"C:\Users\KULDIP\OneDrive\Desktop\Bluepixel\DL\YOLO\Num_plate\test\images\11-Guerrero-2BPlaca-2Bcapacidades-2Bdiferentes-2B99-GAA-2BFranja-2Babajo_jpg.rf.25db474acc416e3ddf70e0080083ac7d.jpg"
image = cv2.imread(image_path)

# 2. Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# 3. Initialize OCR
reader = Reader(['en', 'hi'])   # English + Hindi

# 4. OCR
results = reader.readtext(gray)

# 5. Output
print("\n--- OCR Output ---\n")
for (bbox, text, prob) in results:
    print(f"Text: {text} | Confidence: {prob:.2f}")
