
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# --- Data Preparation ---

# Define the path to your dataset
dataset_path = '/content/drive/MyDrive/YI_IDS4'

# Define image dimensions
img_height = 224
img_width = 224

# Define the path to the training data
train_dataset_path = os.path.join(dataset_path, 'TRAINING')

# Load and preprocess data
images = []
labels = []
class_names = sorted([d for d in os.listdir(train_dataset_path) if os.path.isdir(os.path.join(train_dataset_path, d))])

for class_name in class_names:
    class_path = os.path.join(train_dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        if os.path.isfile(img_path): # Check if it is a file
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_width, img_height))
                img_array = np.asarray(img)
                images.append(img_array)
                labels.append(class_names.index(class_name))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

images = np.array(images)
labels = np.array(labels)

# Normalize images
images = (images.astype(np.float32) / 127.5) - 1

# Split data into training, validation, and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)
val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42, stratify=test_labels)

print(f"Training images shape: {train_images.shape}")
print(f"Validation images shape: {val_images.shape}")
print(f"Testing images shape: {test_images.shape}")
print(f"Number of training labels: {len(train_labels)}")
print(f"Number of validation labels: {len(val_labels)}")
print(f"Number of testing labels: {len(test_labels)}")
print(f"Class names: {class_names}")


# --- Model Definition ---

# Get the number of classes from the preprocessed data
num_classes = len(class_names)

# Define the model architecture
# Using a pre-trained MobileNetV2 model as a base
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top of the pre-trained base
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Display the model summary
model.summary()

# --- Model Compilation ---

# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Display the model summary after compilation
model.summary()

# --- Model Training ---

# Train the model
epochs = 10
history = model.fit(train_images, train_labels,
                    epochs=epochs,
                    validation_data=(val_images, val_labels))

# --- Model Evaluation (Optional - can be done in a separate script) ---
# test_results = model.evaluate(test_images, test_labels, verbose=0)
# print(f"Test Loss: {test_results[0]:.4f}")
# print(f"Test Accuracy: {test_results[1]:.4f}")

# --- Model Saving ---
model.save('my_trained_model.h5')
