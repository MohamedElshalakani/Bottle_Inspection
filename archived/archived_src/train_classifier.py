import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # or whatever classifier you’re using
import joblib

# 1️⃣ Paths and categories
base_path = "../Images"  # adjust if needed
categories = [
    "Good Bottles",
    "Defective Bottles - Missing Cap",
    "Defective Bottles - No Label",
]

# 2️⃣ Load images
X = []
y = []

for idx, cat in enumerate(categories):
    folder = os.path.join(base_path, cat, "Training Sample").replace("\\", "/")
    if not os.path.exists(folder):
        print(f"Warning: Folder not found -> {folder}")
        continue

    files = os.listdir(folder)
    print(f"{cat}: {len(files)} images")  # check count

    for file in files:
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        img = cv2.resize(img, (200, 200))
        X.append(img)
        y.append(idx)

# 3️⃣ Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("Data loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# 4️⃣ Flatten images if using SVM
X_flat = X.reshape(len(X), -1)

# 5️⃣ Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train), "Testing samples:", len(X_test))

# 6️⃣ Train classifier
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# 7️⃣ Test accuracy
acc = clf.score(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(acc * 100))

# 8️⃣ Save model
joblib.dump(clf, "bottle_classifier.pkl")
print("Classifier saved as bottle_classifier.pkl")
print("Checking folder:", folder)
print("Exists?", os.path.exists(folder))
