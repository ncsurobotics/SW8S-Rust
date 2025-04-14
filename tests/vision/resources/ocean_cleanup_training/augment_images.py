import os
import cv2
import albumentations as A
from albumentations import (
    Rotate, HorizontalFlip, VerticalFlip, RandomBrightnessContrast,
    RandomScale, GaussNoise, MotionBlur, Compose
)

# Input & output folders
#replace this with the image folder you want to augment 
input_folder = r'C:\Users\chris\Documents\learning\ncstate\underwater-robotics\SW8S-Rust\tests\vision\resources\ocean_cleanup_training\sawfish_images_augmented'
output_folder = r'C:\Users\chris\Documents\learning\ncstate\underwater-robotics\SW8S-Rust\tests\vision\resources\ocean_cleanup_training\sawfish_images_augmented'


os.makedirs(output_folder, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    Rotate(limit=20, p=0.9),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.2),
    RandomBrightnessContrast(p=0.8),
    RandomScale(scale_limit=0.1, p=0.5),
    GaussNoise(p=0.3),
    MotionBlur(blur_limit=5, p=0.2),
])

num_augmented_images = 5

for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Could not read {filename}")
        continue

    # Save original
    cv2.imwrite(os.path.join(output_folder, f"original_{filename}"), image)

    # Augment images
    for i in range(num_augmented_images):
        augmented = transform(image=image)
        aug_img = augmented['image']
        new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, new_filename), aug_img)

print("Augmentation Complete!")
