import os

train_path = "dataset/Tomato/train"
val_path = "dataset/Tomato/val"

print("=" * 60)
print("TRAIN DATASET")
print("=" * 60)

total_train = 0

for folder in sorted(os.listdir(train_path)):
    folder_path = os.path.join(train_path, folder)

    if os.path.isdir(folder_path):
        image_count = len(os.listdir(folder_path))
        total_train += image_count
        print(f"{folder:<45} {image_count}")

print("\nTotal Training Images:", total_train)

print("\n" + "=" * 60)
print("VALIDATION DATASET")
print("=" * 60)

total_val = 0

for folder in sorted(os.listdir(val_path)):
    folder_path = os.path.join(val_path, folder)

    if os.path.isdir(folder_path):
        image_count = len(os.listdir(folder_path))
        total_val += image_count
        print(f"{folder:<45} {image_count}")

print("\nTotal Validation Images:", total_val)