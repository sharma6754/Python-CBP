import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import seaborn as sns

# Set image resizing dimensions
IMG_SIZE = (64, 64)

# Helper function to read and resize images
read = lambda imname: np.asarray(Image.open(imname).resize(IMG_SIZE).convert("RGB"))

# Load images from a folder
def load_images(folder):
    if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
        print(f"Warning: Folder '{folder}' is empty or does not exist.")
        return np.empty((0, *IMG_SIZE, 3), dtype='uint8')
    return np.array([read(os.path.join(folder, filename)) for filename in os.listdir(folder)], dtype='uint8')

# Paths
folder_benign_train = 'train/benign'
folder_malignant_train = 'train/malignant'
folder_benign_test = 'test/benign'
folder_malignant_test = 'test/malignant'

# Load datasets
X_benign = load_images(folder_benign_train)
X_malignant = load_images(folder_malignant_train)
X_benign_test = load_images(folder_benign_test)
X_malignant_test = load_images(folder_malignant_test)

# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])
y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

# Merge datasets
X_train = np.concatenate((X_benign, X_malignant), axis=0)
y_train = np.concatenate((y_benign, y_malignant), axis=0)
X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)

# Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train, y_train = X_train[s], y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test, y_test = X_test[s], y_test[s]

# Normalize data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Flatten and reduce dimensionality using PCA
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

pca = PCA(n_components=0.95)  # Keep 95% of the variance
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# Train SVM model
model = SVC()
model.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy}")

# Save the trained model, PCA, and accuracy
joblib.dump(model, 'svm_skin_cancer_model.pkl')
joblib.dump(pca, 'pca_skin_cancer.pkl')
joblib.dump(accuracy, 'svm_skin_cancer_accuracy.pkl')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
joblib.dump(cm, 'confusion_matrix.pkl')

# Optionally, plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Model, PCA, accuracy, and confusion matrix saved successfully!")

# --- Predict for a User-Specified Image ---
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("Error: The specified image path does not exist.")
        return

    # Load and preprocess the image
    img = read(image_path)
    img_normalized = img.astype('float32') / 255  # Normalize
    img_flat = img_normalized.reshape(1, -1)     # Flatten
    img_pca = pca.transform(img_flat)            # Apply PCA

    # Make prediction
    prediction = model.predict(img_pca)[0]
    label = "Malignant" if prediction == 1 else "Benign"

    # Display the result
    print(f"The model predicts that the given image is: {label}")
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()

# Example usage:
# Provide the path to the image you want to predict
predict_image('38347tn.jpg')
