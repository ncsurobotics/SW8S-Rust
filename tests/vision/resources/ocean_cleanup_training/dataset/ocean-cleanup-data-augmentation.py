import os
import cv2
import albumentations as A

# Input
INPUT_IMG_DIR = 'images'
INPUT_LABEL_DIR = 'labels'

# Output
OUTPUT_IMG_DIR = 'images/train'
OUTPUT_LABEL_DIR = 'labels/train'

# Create output directories
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=20, p=0.5),
    A.MotionBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.Resize(640, 640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

AUG_PER_IMAGE = 5

# Loop through images
for filename in os.listdir(INPUT_IMG_DIR):
    if not filename.endswith('.png'):
        continue

    image_path = os.path.join(INPUT_IMG_DIR, filename)
    label_path = os.path.join(INPUT_LABEL_DIR, filename.replace('.png', '.txt'))

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    # Load YOLO boxes
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

    # Save original to new folder if you want to include them in training
    orig_out_img = os.path.join(OUTPUT_IMG_DIR, filename)
    orig_out_lbl = os.path.join(OUTPUT_LABEL_DIR, filename.replace('.png', '.txt'))
    cv2.imwrite(orig_out_img, image)
    with open(orig_out_lbl, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

    # Create augmented versions
    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']

        new_img_name = filename.replace('.png', f'_aug_{i}.png')
        new_lbl_name = new_img_name.replace('.png', '.txt')

        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_img_name), aug_img)

        with open(os.path.join(OUTPUT_LABEL_DIR, new_lbl_name), 'w') as f:
            for cls, bbox in zip(class_labels, aug_bboxes):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
