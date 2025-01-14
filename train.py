import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set paths
dataset_dir = "dataset"  # Change to your dataset's path

# Parameters
img_height = 224
img_width = 224
batch_size = 32
num_classes = 4

# Data preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
)

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    shuffle=True,
)

# Model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=10,  # Adjust as necessary
    validation_data=train_data  # Use the same dataset for validation
)

# Save the model
model.save("material_classifier.h5")

# Evaluate the model
loss, accuracy = model.evaluate(train_data)
print(f"Final Loss: {loss}")
print(f"Final Accuracy: {accuracy}")

# Class indices mapping
print("Class indices:", train_data.class_indices)
