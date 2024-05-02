import numpy as np
import os
import PIL
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fileProcessor import load_images_from_directory_train, load_images_from_directory_test
from fileDownloader import training_files, testing_files

# Algorithms to make sure the .DS_Store files get ignored during the analysis stage

'''
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
            img = img.resize((224, 224))
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
        img_path = os.path.join(testing_files_files, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((224, 224))
            x.append(np.array(img))

            label = os.path.basename(testing_files_files)
            y.append(label)

        except(PIL.UnidentifiedImageError, OSError):
            print(f"Skipping {filename}: Not a valid image")

    # Print the number of loaded images
    print(f"Number of loaded images: {len(x)}")

    return np.array(x), np.array(y)
'''


# Preprocessing the image

'''
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image.thumbnail(target_size)
    new_image = Image.new('RGB', target_size)
    new_image.paste(image, ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2))
    return new_image
'''

