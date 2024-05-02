import os
import PIL


# Downloading files used for the analysis

training_files = 'Final Year Project Analysis Files/TRIAL_FOR_TESTING'
testing_files = 'Final Year Project Analysis Files/TRIAL_FOR_TESTING'

# Print out the folder paths to ensure they are visible

print("Files in the folder: {}".format(training_files))
print("Files in the folder: {}".format(testing_files))

# Count images in the folder to ensure all files are inside the folders

def training_image_number(training_files):
    num_images = 0

    try:
        for file in os.listdir(training_files):
            if file.lower().endswith(".png"):
                num_images += 1

        training_images = num_images

        if training_images == 0:
            print("No training images found")
        else:
            print("Number of training images: {}".format(training_images))
    except Exception as error:
        print(f"An error occurred: {error}")


training_image_number(training_files)


def testing_image_number(testing_files):
    num_images = 0

    try:
        for file in os.listdir(testing_files):
            if file.lower().endswith(".png"):
                num_images += 1

        testing_images = num_images

        if testing_images == 0:
            print("No training images found")
        else:
            print("Number of testing images: {}".format(testing_images))
    except Exception as error:
        print(f"An error occurred: {error}")


testing_image_number(testing_files)
