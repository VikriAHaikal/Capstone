
"""# Split Train and Validation"""

import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Specify the paths to original dataset and the destination directory
original_dataset_path = 'fixbanget'
destination_path = 'pythonProject'

# Get a list of all subdirectories in the original dataset path
class_directories = os.listdir(original_dataset_path)
train_split_ratio = 0.8
print(class_directories)

class_dict = {}
for i, class_directory in enumerate(class_directories):
    class_path = os.path.join(original_dataset_path, class_directory)

    # Check if the item in the directory is a directory itself
    if os.path.isdir(class_path):
        class_dict[i] = class_directory
        # Create a destination directory for the class in the new structure
        # Create a destination directory for the class in the new structure
        destination_train_class_path = os.path.join(destination_path, 'train', class_directory)
        os.makedirs(destination_train_class_path, exist_ok=True)

        destination_validation_class_path = os.path.join(destination_path, 'validation', class_directory)
        os.makedirs(destination_validation_class_path, exist_ok=True)

        # Get a list of all image files in the class directory
        image_files = os.listdir(class_path)

        # Shuffle the image files randomly
        random.shuffle(image_files)

        # Calculate the split index based on the split ratio
        split_index = int(train_split_ratio * len(image_files))

        # Split the image files into train and validation sets
        train_files = image_files[:split_index]
        validation_files = image_files[split_index:]

        # Move each train image file to the corresponding class directory in the train split
        for train_file in train_files:
            image_path = os.path.join(class_path, train_file)
            destination_image_path = os.path.join(destination_train_class_path, train_file)
            shutil.copy2(image_path, destination_image_path)

        # Move each validation image file to the corresponding class directory in the validation split
        for validation_file in validation_files:
            image_path = os.path.join(class_path, validation_file)
            destination_image_path = os.path.join(destination_validation_class_path, validation_file)
            shutil.copy2(image_path, destination_image_path)

print('Dataset reorganized and split successfully!')

"""# Data Preprocess"""

PATH = os.path.join("dataset")
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

class_names = train_dataset.class_names
print(class_names)

# Save Classname in txt file
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2),
  tf.keras.layers.RandomContrast(0.2),
  tf.keras.layers.RandomBrightness(0.2),
])

for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./255, offset=-1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

"""# Create and Training Model"""

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.EfficientNetV2M(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(len(class_names))
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

initial_epochs = 100
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')
callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss",
                            factor=0.25,
                            patience=1,
                            verbose=1,
                            min_lr=1e-7
                        )

with tf.device('/GPU:0'):
    history = model.fit(train_dataset,
                      epochs=initial_epochs,
                      validation_data=validation_dataset)

loss, accuracy = model.evaluate(validation_dataset)
print('Test accuracy :', accuracy)

# Plot the training and validation accuracy and loss at each epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save("saved_model/saved_model_format", include_optimizer=False)

(tf. __version__)

model.save("saved_model/model_saved")

tf.saved_model.save(model,"saved_model/h5_format_new" )
