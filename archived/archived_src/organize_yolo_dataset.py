import os
import shutil
import random

# Paths
base_path = os.path.expanduser("~/Bottle_Inspection/Images")
output_path = os.path.expanduser("~/Bottle_Inspection-3")

# Classes
classes = ["Good", "No_Cap", "No_Label"]

# Train/val split
val_ratio = 0.2  # 20% validation

# Create YOLO folder structure
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

print("📂 Organizing dataset...\n")

# Process each class
for cls in classes:
    class_folder = os.path.join(base_path, cls, "Training_Sample")

    if not os.path.exists(class_folder):
        print(f"⚠️ Skipping {cls}, folder not found")
        continue

    images = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]

    if len(images) == 0:
        print(f"⚠️ No images in {cls}")
        continue

    random.shuffle(images)

    split_index = int(len(images) * (1 - val_ratio))
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"➡️ {cls}: {len(train_images)} train, {len(val_images)} val")

    # Copy images
    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        for img in split_images:
            src = os.path.join(class_folder, img)
            dst = os.path.join(output_path, split_name, "images", img)

            # Avoid overwrite (just in case)
            if os.path.exists(dst):
                continue

            shutil.copy(src, dst)

print("\n✅ Dataset organized successfully!")
print(f"📁 Output folder: {output_path}")