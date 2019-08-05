import os
import shutil
from PIL import Image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from collections import defaultdict


class BhatkhandeClassifier:
    """
    Attributes:
        _num_total_images: the number of images to generate for each sample
        _num_training_images: the number of images used for training for each sample
        _num_validation_images: the number of images used for validation for each sample
        _input_dir: the directory in which the samples are stored
        _num_classes: the number of classes
        _target_image_size: the target width/height of an image
        _num_channels: the number of channels in the sample images
        _batch_size: the batch size
        _num_epochs: the number of epochs
        _model: the model

    Methods:
        generate_images: generates images for each sample
        move_generated_images: moves the generated images to either the train or validation folder
        train_model: trains the model
        save_model: saves the model
    """

    def __init__(self, num_total_images, percentage_train, input_dir, num_epochs):
        """BhatkhandeClassifier constructor."""
        self._num_total_images = num_total_images
        self._num_training_images = num_total_images * percentage_train
        self._num_validation_images = num_total_images * (1 - percentage_train)
        self._input_dir = input_dir
        self._num_classes = self._calculate_num_classes()
        self._target_image_size = 100
        self._num_channels = 3
        self._batch_size = self._num_total_images // 10
        self._num_epochs = num_epochs
        self._model = Sequential()
        self._set_up_file_structure()

    def _calculate_num_classes(self):
        """Calculate the number of class in the directory."""
        directory = os.path.join(os.path.dirname(__file__), self._input_dir)
        valid_file_extension = ".png"
        num_classes = 0

        for file in os.listdir(directory):
            if file.endswith(valid_file_extension):
                num_classes += 1

        return num_classes

    @staticmethod
    def _create_directory(directory):
        """Create directory."""
        if os.path.isdir(directory):
            print("Removing {}...".format(directory))
            shutil.rmtree(directory)

        print("Creating {}...".format(directory))
        os.makedirs(directory)

    @staticmethod
    def _create_subdirectories_in(directory, source):
        """Create subdirectories in directory."""
        for file in os.listdir(source):
            if file.endswith(".png"):
                print("Creating {}...".format(os.path.join(directory, file.split('.')[0])))
                os.makedirs(os.path.join(directory, file.split('.')[0]))

    @staticmethod
    def _remove_saved_model(path):
        """Remove the file at the given path."""
        if os.path.isfile(path):
            print("Removing {}...".format(path))
            os.remove(path)

    def _set_up_file_structure(self):
        """Set up the file structure."""
        self._create_directory(directory=os.path.join(os.path.dirname(__file__), "train"))
        self._create_directory(directory=os.path.join(os.path.dirname(__file__), "validation"))
        self._create_directory(directory=os.path.join(os.path.dirname(__file__), "generated_images"))
        self._create_subdirectories_in(directory=os.path.join(os.path.dirname(__file__), "train"),
                                       source=os.path.join(os.path.dirname(__file__), self._input_dir))
        self._create_subdirectories_in(directory=os.path.join(os.path.dirname(__file__), "validation"),
                                       source=os.path.join(os.path.dirname(__file__), self._input_dir))
        self._remove_saved_model(path=os.path.join(os.path.dirname(__file__), "model.json"))
        self._remove_saved_model(path=os.path.join(os.path.dirname(__file__), "model.h5"))

    def _pad_image(self, path):
        """Pad the image."""
        image = Image.open(path)
        current_image_width, current_image_height = image.size
        target_image_side_length = max(current_image_height, current_image_width)
        background = self._get_background(target_image_side_length=target_image_side_length)
        background_offset = self._get_background_offset(target_image_side_length=target_image_side_length,
                                                        current_image_width=current_image_width,
                                                        current_image_height=current_image_height)
        background.paste(image, background_offset)
        background.save(path)

    @staticmethod
    def _get_background(target_image_side_length):
        """Get the background."""
        background_color_mode = "RGBA"
        background_size = (target_image_side_length, target_image_side_length)
        background_color = (255, 255, 255, 255)
        background = Image.new(background_color_mode, background_size, background_color)
        return background

    @staticmethod
    def _get_background_offset(target_image_side_length, current_image_width, current_image_height):
        """Get the background offset."""
        x_coord = round((target_image_side_length - current_image_width) / 2)
        y_coord = round((target_image_side_length - current_image_height) / 2)
        return x_coord, y_coord

    def _resize_image(self, path):
        """Resize the image."""
        image = Image.open(path)
        maximum_dimensions = (self._target_image_size, self._target_image_size)
        resized_image = image.resize(maximum_dimensions)
        resized_image.save(path)

    def generate_images(self):
        """Generate _num_total_images images for each sample and save them to generated_images."""
        print("Generating the images...")
        data_generator = self._get_image_data_generator()
        directory = os.path.join(os.path.dirname(__file__), self._input_dir)
        valid_file_extension = ".png"

        for file in os.listdir(directory):
            if file.endswith(valid_file_extension):
                file_path = os.path.join(directory, file)
                self._pad_image(path=file_path)
                self._resize_image(path=file_path)
                image = img_to_array(load_img(file_path))
                image = image.reshape((1,) + image.shape)
                character_name = file.split('.')[0]
                num_generated_images = 0

                for _ in data_generator.flow(image,
                                             batch_size=1,
                                             save_to_dir=os.path.join(os.path.dirname(__file__), "generated_images"),
                                             save_prefix=character_name,
                                             save_format="png"):
                    num_generated_images += 1

                    if num_generated_images >= self._num_total_images:
                        break

    @staticmethod
    def _get_image_data_generator():
        """Get the image data generator."""
        return ImageDataGenerator(
            rotation_range=1,
            width_shift_range=0.02,
            height_shift_range=0.02,
            rescale=1 / 255,
            shear_range=0.05,
            zoom_range=0.05,
            brightness_range=(0, 2),
            fill_mode="nearest")

    def move_generated_images(self):
        """Move the images in generated_images to the train and validation directories."""
        print("Moving the generated images to train and validation...")
        group_counts = defaultdict(int)
        generated_images_directory = os.path.join(os.path.dirname(__file__), "generated_images")
        valid_file_extension = ".png"
        train_directory = os.path.join(os.path.dirname(__file__), "train")
        validation_directory = os.path.join(os.path.dirname(__file__), "validation")

        for file in os.listdir(generated_images_directory):
            if file.endswith(valid_file_extension):
                group = file.split('_')[0]
                group_counts[group] += 1

                if group_counts[group] <= self._num_training_images:
                    shutil.move(os.path.join(generated_images_directory, file), os.path.join(train_directory, group))
                else:
                    shutil.move(os.path.join(generated_images_directory, file),
                                os.path.join(validation_directory, group))

        shutil.rmtree(generated_images_directory)

    def train_model(self):
        """Train the model."""
        print("Training the model...")
        train_directory = os.path.join(os.path.dirname(__file__), "train")
        validation_directory = os.path.join(os.path.dirname(__file__), "validation")
        target_size = (self._target_image_size, self._target_image_size)
        color_mode = "rgb"
        class_mode = "categorical"
        data_generator = ImageDataGenerator(rescale=1 / 255)

        train_generator = data_generator.flow_from_directory(directory=train_directory,
                                                             target_size=target_size,
                                                             color_mode=color_mode,
                                                             batch_size=self._batch_size,
                                                             class_mode=class_mode)

        validation_generator = data_generator.flow_from_directory(directory=validation_directory,
                                                                  target_size=target_size,
                                                                  color_mode=color_mode,
                                                                  batch_size=self._batch_size,
                                                                  class_mode=class_mode)

        self._model.add(Conv2D(filters=32,
                               kernel_size=(3, 3),
                               activation="relu",
                               input_shape=(self._target_image_size, self._target_image_size, self._num_channels)))
        self._model.add(Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation="relu"))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))
        self._model.add(Flatten())
        self._model.add(Dense(units=128,
                              activation="relu"))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(units=self._num_classes,
                              activation="softmax"))

        self._model.compile(loss=categorical_crossentropy,
                            optimizer=Adadelta(),
                            metrics=["accuracy"])

        self._model.fit_generator(generator=train_generator,
                                  steps_per_epoch=self._num_training_images // self._batch_size,
                                  validation_data=validation_generator,
                                  validation_steps=self._num_validation_images // self._batch_size,
                                  epochs=self._num_epochs,
                                  verbose=2)

    def save_model(self):
        """Save the model."""
        print("Saving the model...")
        with open(os.path.join(os.path.dirname(__file__), "model.json"), 'w') as file:
            file.write(self._model.to_json())
        self._model.save_weights(os.path.join(os.path.dirname(__file__), "model.h5"))


def main():
    """Main function."""
    bc = BhatkhandeClassifier(num_total_images=100,
                              percentage_train=0.8,
                              input_dir="samples",
                              num_epochs=12)
    bc.generate_images()
    bc.move_generated_images()
    bc.train_model()
    bc.save_model()


if __name__ == "__main__":
    main()
