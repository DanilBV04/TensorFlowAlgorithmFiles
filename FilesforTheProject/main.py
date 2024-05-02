import numpy as np
import os
import PIL
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from fileDownloader import training_files, testing_files
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from flask import Flask, request, jsonify
# Adding the preprocessing layer for the images, so that they can be adjusted beforehand


# Copy and Pasting module from fileProcessor to try and fix an issue

def load_images_from_directory_train(training_files):
    x = []
    y = []

    # Print the number of files found in the directory
    files = os.listdir(training_files)
    print(f"Number of files in directory: {len(files)}")

    for filename in files:
        if filename.endswith('.DS_Store'):
            continue
        img_path = os.path.join(training_files, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((400, 400))
            x.append(np.array(img))

            label = os.path.basename(training_files)
            y.append(label)

        except(PIL.UnidentifiedImageError, OSError):
            print(f"Skipping {filename}: Not a valid image")

    # Print the number of loaded images
    print(f"Number of loaded images: {len(x)}")

    return np.array(x), np.array(y)


def load_images_from_directory_test(testing_files):
    x = []
    y = []

    # Print the number of files found in the directory
    files = os.listdir(testing_files)
    print(f"Number of files in directory: {len(files)}")

    for filename in files:
        if filename.endswith('.DS_Store'):
            continue
        img_path = os.path.join(testing_files, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((400, 400))
            x.append(np.array(img))

            label = os.path.basename(testing_files)
            y.append(label)

        except(PIL.UnidentifiedImageError, OSError):
            print(f"Skipping {filename}: Not a valid image")

    # Print the number of loaded images
    print(f"Number of loaded images: {len(x)}")

    return np.array(x), np.array(y)


# Loading in the image files in to the correct places for analysis

x_train, y_train = load_images_from_directory_train(training_files)

x_test, y_test = load_images_from_directory_test(testing_files)

# Generating the data needed

train_dataGeneration = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = train_dataGeneration.flow_from_directory(
    training_files,
    target_size=(400, 400),
    batch_size=128,
    class_mode='binary'
)

num_classes = len(np.unique(y_train))

# Encoding labels for better program performance

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print("Encoded labels: ", y_train_encoded)

test_dataGeneration = ImageDataGenerator(rescale=1./255)

test_gen = test_dataGeneration.flow_from_directory(
    testing_files,
    target_size=(400, 400),
    batch_size=32,
    class_mode='binary'
)


# Trying out adding different weights to classes to increase the accuracy of the test


unique_classes, class_counts = np.unique(train_gen.classes, return_counts=True)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(unique_classes) * class_counts)
class_weight_dict = dict(zip(unique_classes, class_weights))

# Adding check for dictionary to cover all class indices

max_class_index = np.max(train_gen.classes)
for i in range(max_class_index + 1):
    if i not in class_weight_dict:
        class_weight_dict[i] = 1.0

# Building a personally edited learning rate, to evade the accuracy and loss scores being invalid

learning_rate = 0.001

# Creating an instance of Adam optimizer, where the learning rate is adjusted to personal preferences

optimizer = Adam(learning_rate=learning_rate)

# Building the analysis model

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(60, activation='sigmoid', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.000001)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Fitting the model against itself to retrieve accurate accuracy results


# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(
    train_gen,
    epochs=25,
    validation_data=test_gen,
    callbacks=[reduce_lr],
    class_weight=class_weight_dict
)

# Inspecting for any misclassified samples

predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
misclassified_indices = np.where(predicted_classes != true_classes)[0]

# Plotting training and validation accuracy values for a better understanding

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plotting training and validation loss values on the graph

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

# Plotting misclassified files
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_indices[:25]):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_gen[0][0][0])
    plt.title(f"Predicted: {predicted_classes[idx]}, True: {true_classes[idx]}")
    plt.axis('off')
plt.show()

test_loss, test_acc = model.evaluate(test_gen)
print("Test accuracy:", test_acc)

