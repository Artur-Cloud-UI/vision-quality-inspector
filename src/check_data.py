import os
import cv2

data_dir = "data"
train_dir = os.path.join(data_dir, "train", "good")
print(f"Checking images in {train_dir}...")
if not os.path.exists(train_dir):
    print(f"Directory {train_dir} does not exist.")
else:
    good_images = os.listdir(train_dir)
    print(f"Found {len(good_images)} images in {train_dir}.")
    if len(good_images) > 0:
        sample_path = os.path.join(train_dir, good_images[0])
        img = cv2.imread(sample_path)
        print(f"Sample image shape: {img.shape}")