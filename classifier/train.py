import os
import shutil
import classifier.image_util
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class BhatkhandeClassifier:
    """
    Attributes:
        __valid_file_extension: the valid file extension for images
        __num_images_per_sample: the number of images to generate for each sample
        __num_training_images: the number of images used for training for each sample
        __num_validation_images: the number of images used for validation for each sample
        __input_dir_path: the directory path in which the samples are stored
        __generated_images_dir_path: the directory path in which the generated images are stored
        __train_dir_path: the directory path in which the images used for training are stored
        __validation_dir_path: the directory path in which the images used for validation are stored
        __model_json_path: the path of the model.json file
        __model_h5_path: the path of the model.h5 file
        __num_classes: the number of classes
        __target_image_size: the target width/height of an image
        __num_channels: the number of channels in the sample images
        __batch_size: the batch size
        __num_epochs: the number of epochs
        __model: the model

    Methods:
        generate_images: generates images for each sample
        move_generated_images: moves the generated images to either the train or validation folder
        train_model: trains the model
        save_model: saves the model
        plot_history: plots the history
    """

    def __init__(self, num_images_per_sample, percentage_train, input_dir, num_epochs):
        """
        BhatkhandeClassifier constructor.
        :param num_images_per_sample: an int
        :param percentage_train: a float
        :param input_dir: a string
        :param num_epochs: an int
        """
        self.__valid_file_extension = ".png"
        self.__num_images_per_sample = num_images_per_sample
        self.__num_training_images = round(num_images_per_sample * percentage_train)
        self.__num_validation_images = round(num_images_per_sample * (1 - percentage_train))
        self.__input_dir_path = os.path.join(os.path.dirname(__file__), input_dir)
        self.__generated_images_dir_path = os.path.join(os.path.dirname(__file__), "generated_images")
        self.__train_dir_path = os.path.join(os.path.dirname(__file__), "train")
        self.__validation_dir_path = os.path.join(os.path.dirname(__file__), "validation")
        self.__model_json_path = os.path.join(os.path.dirname(__file__), "model.json")
        self.__model_h5_path = os.path.join(os.path.dirname(__file__), "model.h5")
        self.__num_classes = self.__get_num_classes()
        self.__target_image_size = 30
        self.__num_channels = 3
        self.__batch_size = self.__num_images_per_sample // 10
        self.__num_epochs = num_epochs
        self.__model = Sequential()
        self.__history = None
        self.__verify_file_names()
        self.__set_up_file_structure()

    def __verify_file_names(self):
        """
        Verify file names.
        :return: nothing
        """
        for element in os.listdir(self.__input_dir_path):
            if os.path.isdir(os.path.join(self.__input_dir_path, element)):
                for file in os.listdir(os.path.join(self.__input_dir_path, element)):
                    file_without_extension = file.split(".")[0]

                    if "_" in file_without_extension:
                        raise Exception("A file name must not contain underscores")
                    elif len(file_without_extension.split("-")) < 2:
                        raise Exception("A file name must be split by hyphens")
                    else:
                        file_without_extension_parts = file_without_extension.split("-")

                        if not file_without_extension_parts[-1].isdigit():
                            raise Exception("The last part of a file name must be an integer")
            else:
                raise Exception("The {} directory must only contain directories".format(self.__input_dir_path))

    def __get_num_classes(self):
        """
        Get the number of classes.
        :return: an int
        """
        num_classes = 0

        for element in os.listdir(self.__input_dir_path):
            if os.path.isdir(os.path.join(self.__input_dir_path, element)):
                num_classes += 1

        return num_classes

    @staticmethod
    def __create_directory(directory):
        """
        Create the directory.
        :param directory: a string
        :return: nothing
        """
        if os.path.isdir(directory):
            print("Removing {}...".format(directory))
            shutil.rmtree(directory)

        print("Creating {}...".format(directory))
        os.makedirs(directory)

    def __create_subdirectories(self, directory):
        """
        Create subdirectories in the directory.
        :param directory: a string
        :return: nothing
        """
        for element in os.listdir(self.__input_dir_path):
            if os.path.isdir(os.path.join(self.__input_dir_path, element)):
                print("Creating {}...".format(os.path.join(directory, element)))
                os.makedirs(os.path.join(directory, element))

    @staticmethod
    def __remove_file(path):
        """
        Remove the file at the given path.
        :param path: a string
        :return: nothing
        """
        if os.path.isfile(path):
            print("Removing {}...".format(path))
            os.remove(path)

    def __set_up_file_structure(self):
        """
        Set up the file structure.
        :return: nothing
        """
        self.__create_directory(directory=self.__generated_images_dir_path)
        self.__create_directory(directory=self.__train_dir_path)
        self.__create_directory(directory=self.__validation_dir_path)
        self.__create_subdirectories(directory=self.__train_dir_path)
        self.__create_subdirectories(directory=self.__validation_dir_path)
        self.__remove_file(path=self.__model_json_path)
        self.__remove_file(path=self.__model_h5_path)

    def generate_images(self):
        """
        Generate images.
        :return: nothing
        """
        print("Generating the images...")
        data_generator = self.__get_image_data_generator()

        for element in os.listdir(self.__input_dir_path):
            if os.path.isdir(os.path.join(self.__input_dir_path, element)):
                for file in os.listdir(os.path.join(self.__input_dir_path, element)):
                    if file.endswith(self.__valid_file_extension):
                        path = os.path.join(self.__input_dir_path, element, file)
                        classifier.image_util.pad_image(path=path)
                        classifier.image_util.resize_image(path=path)
                        image = classifier.image_util.reshape_image(path=path)
                        file_without_extension = file.split('.')[0]
                        num_generated_images = 0

                        for _ in data_generator.flow(image,
                                                     batch_size=1,
                                                     save_to_dir=self.__generated_images_dir_path,
                                                     save_prefix=file_without_extension,
                                                     save_format="png"):
                            num_generated_images += 1

                            if num_generated_images >= self.__num_images_per_sample:
                                break

    @staticmethod
    def __get_image_data_generator():
        """
        Get the image data generator.
        :return: an ImageDataGenerator
        """
        return ImageDataGenerator(rotation_range=1,
                                  width_shift_range=0.02,
                                  height_shift_range=0.02,
                                  rescale=1 / 255,
                                  shear_range=0.05,
                                  zoom_range=0.05,
                                  brightness_range=(0, 2),
                                  fill_mode="nearest")

    def move_generated_images(self):
        """
        Move the generated images.
        :return: nothing
        """
        print("Moving the generated images to train and validation...")
        character_counts = defaultdict(int)

        for file in os.listdir(self.__generated_images_dir_path):
            if file.endswith(self.__valid_file_extension):
                original_file_without_extension = file.split(sep='_')[0]
                character = original_file_without_extension.rsplit(sep='-', maxsplit=1)[0]
                character_counts[original_file_without_extension] += 1
                src_file_path = os.path.join(self.__generated_images_dir_path, file)
                
                if character_counts[original_file_without_extension] <= self.__num_training_images:
                    train_dir_group_path = os.path.join(self.__train_dir_path, character)
                    shutil.move(src_file_path, train_dir_group_path)
                else:
                    validation_dir_group_path = os.path.join(self.__validation_dir_path, character)
                    shutil.move(src_file_path, validation_dir_group_path)

        shutil.rmtree(self.__generated_images_dir_path)

    def train_model(self):
        """
        Train the model.
        :return: nothing
        """
        print("Training the model...")
        target_size = (self.__target_image_size, self.__target_image_size)
        color_mode = "rgb"
        class_mode = "categorical"
        data_generator = ImageDataGenerator(rescale=1 / 255)

        train_generator = data_generator.flow_from_directory(directory=self.__train_dir_path,
                                                             target_size=target_size,
                                                             color_mode=color_mode,
                                                             batch_size=self.__batch_size,
                                                             class_mode=class_mode)

        validation_generator = data_generator.flow_from_directory(directory=self.__validation_dir_path,
                                                                  target_size=target_size,
                                                                  color_mode=color_mode,
                                                                  batch_size=self.__batch_size,
                                                                  class_mode=class_mode)

        self.__model.add(Conv2D(filters=32,
                                kernel_size=(3, 3),
                                activation="relu",
                                input_shape=(self.__target_image_size, self.__target_image_size, self.__num_channels)))
        self.__model.add(Conv2D(filters=64,
                                kernel_size=(3, 3),
                                activation="relu"))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Dropout(0.25))
        self.__model.add(Flatten())
        self.__model.add(Dense(units=128,
                               activation="relu"))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(units=self.__num_classes,
                               activation="softmax"))

        self.__model.compile(loss=categorical_crossentropy,
                             optimizer=Adadelta(),
                             metrics=["accuracy"])

        self.__history = self.__model.fit_generator(generator=train_generator,
                                                    steps_per_epoch=self.__num_training_images // self.__batch_size,
                                                    validation_data=validation_generator,
                                                    validation_steps=self.__num_validation_images // self.__batch_size,
                                                    epochs=self.__num_epochs,
                                                    verbose=2)

    def save_model(self):
        """
        Save the model.
        :return: nothing
        """
        print("Saving the model...")
        with open(self.__model_json_path, 'w') as file:
            file.write(self.__model.to_json())
        self.__model.save_weights(self.__model_h5_path)

    def plot_history(self):
        """
        Plot the history.
        :return: nothing
        """
        epochs = range(1, self.__num_epochs + 1)
        training_loss = self.__history.history["loss"]
        validation_loss = self.__history.history["val_loss"]
        training_accuracy = self.__history.history["acc"]
        validation_accuracy = self.__history.history["val_acc"]
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.suptitle("Training History")
        ax1.plot(epochs, training_loss)
        ax1.plot(epochs, validation_loss)
        ax1.set_ylabel("Loss")
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.legend(["Training", "Validation"], loc="upper right", title="Dataset:")
        ax2.plot(epochs, training_accuracy)
        ax2.plot(epochs, validation_accuracy)
        ax2.set_ylabel("Accuracy")
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.legend(["Training", "Validation"], loc="lower right", title="Dataset:")
        plt.xlabel("Epoch")
        plt.show()


def main():
    """
    Main function.
    :return: nothing
    """
    try:
        bhatkhande_classifier = BhatkhandeClassifier(num_images_per_sample=200,
                                                     percentage_train=0.8,
                                                     input_dir="new_samples",
                                                     num_epochs=15)
        bhatkhande_classifier.generate_images()
        bhatkhande_classifier.move_generated_images()
        bhatkhande_classifier.train_model()
        bhatkhande_classifier.save_model()
        bhatkhande_classifier.plot_history()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
