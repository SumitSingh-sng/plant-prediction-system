import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Helper function to convert images to arrays
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            print(f"Invalid image: {image_dir}")
            return np.array([])
    except Exception as e:
        print(f"Error : {e} for image {image_dir}")
        return None

# Path to the dataset
dir = "../input/leaf-image-dataset/Plant_images"
root_dir = listdir(dir)

# Labels and binary mappings
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
binary_labels = {label: idx for idx, label in enumerate(all_labels)}

# Preparing image data and labels
image_list, label_list = [], []

for directory in root_dir:
    if directory in all_labels:
        plant_image_list = listdir(f"{dir}/{directory}")
        for files in plant_image_list:
            image_path = f"{dir}/{directory}/{files}"
            img_array = convert_image_to_array(image_path)
            if img_array.size > 0:
                image_list.append(img_array)
                label_list.append(binary_labels[directory])

# Convert lists to arrays
image_list = np.array(image_list, dtype=np.float32) / 255.0  # Normalize pixel values
label_list = np.array(label_list)

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    image_list, label_list, test_size=0.2, random_state=10
)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=len(all_labels))
y_test = to_categorical(y_test, num_classes=len(all_labels))

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(16, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(all_labels), activation="softmax")
])

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# Split training data into training and validation datasets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=10
)

# Train the model
epochs = 50
batch_size = 128
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='r')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Evaluate the model
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {scores[1] * 100:.2f}%")

# Test prediction example
img = array_to_img(x_test[0])
plt.imshow(img)
plt.title("Sample Test Image")
plt.show()

print("Originally:", all_labels[np.argmax(y_test[0])])
print("Predicted:", all_labels[np.argmax(model.predict(np.expand_dims(x_test[0], axis=0)))])