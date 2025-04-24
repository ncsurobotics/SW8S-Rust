import cv2

image_path = 'images/train/sawfish_008_aug_2.png'
label_path = 'labels/train/sawfish_008_aug_2.txt'

# Load image
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Load label
with open(label_path, 'r') as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        # Convert YOLO to pixel box
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"class {int(cls)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Save instead of showing
cv2.imwrite("sawfish_008_aug_2.png", img)
print("[✅] Image saved as sawfish_008_aug_2.png")
