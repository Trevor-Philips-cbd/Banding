import os
import shutil
import random

# Paths for the train and val folders
train_images_dir = '../trains/train/images'
train_labels_dir = '../trains/train/labels'
val_images_dir = '../vals/val/images'
val_labels_dir = '../vals/val/labels'

# Create val directories if they do not exist
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List all image files in the train images directory
image_files = os.listdir(train_images_dir)

# Randomly select 20% of the images
num_val_images = int(len(image_files) * 0.2)
val_image_files = random.sample(image_files, num_val_images)

# Move selected images and their corresponding labels to the val folder
for image_file in val_image_files:
    image_path = os.path.join(train_images_dir, image_file)
    label_path = os.path.join(train_labels_dir, image_file.replace('.png', '.txt'))  # Assuming labels have a corresponding .txt file
    
    # Move images
    shutil.move(image_path, os.path.join(val_images_dir, image_file))
    
    # Move corresponding labels
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(val_labels_dir, os.path.basename(label_path)))

print(f"Moved {num_val_images} images and corresponding labels to val folder.")
