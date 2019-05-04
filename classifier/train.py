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
        _image_size: the maximum width/height of the sample images
        _num_channels: the number of channels in the sample images
        _batch_size: the batch size
        _num_classes: the number of classes
        _num_epochs: the number of epochs
        _model: the model
        _input_dir: the directory in which the samples are stored

    Methods:
        generate_images: generates images for each sample
        move_generated_images: moves the generated images to either the train or validation folder
        train_model: trains the model
        save_model: saves the model
    """

    def __init__(self, num_images_per_sample, percentage_train, samples_directory):
        """BhatkhandeClassifier constructor."""
        self._num_total_images = num_images_per_sample
        self._num_training_images = num_images_per_sample * percentage_train
        self._num_validation_images = num_images_per_sample * (1 - percentage_train)
        self._image_size = 0
        self._num_channels = 3
        self._batch_size = self._num_total_images // 10
        self._num_classes = 5
        self._num_epochs = 5
        self._model = Sequential()
        self._input_dir = samples_directory
        self._set_up_file_structure()

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

    def _resize_image(self, path):
        """Resize the image."""
        image = Image.open(path)
        image_width, image_height = image.size

        if image_width < self._image_size or image_height < self._image_size:
            background = Image.new("RGBA", (self._image_size, self._image_size), (255, 255, 255, 255))
            offset = round((self._image_size - image_width) / 2), round((self._image_size - image_height) / 2)
            background.paste(image, offset)
            background.save(path)

    @staticmethod
    def _scale_image(path):
        """Scale the image."""
        image = Image.open(path)
        maximum_dimensions = (128, 128)
        resized_image = image.resize(maximum_dimensions)
        resized_image.save(path)

    def generate_images(self):
        """Generate _num_total_images images for each sample and save them to generated_images."""
        print("Generating the images...")
        data_generator = ImageDataGenerator(
            rotation_range=1,
            width_shift_range=0.02,
            height_shift_range=0.02,
            rescale=1 / 255,
            shear_range=0.05,
            zoom_range=0.05,
            brightness_range=(0, 2),
            fill_mode="nearest")

        max_width = 0
        max_height = 0

        for file in os.listdir(os.path.join(os.path.dirname(__file__), self._input_dir)):
            if file.endswith(".png"):
                image = Image.open(os.path.join(os.path.join(os.path.dirname(__file__), self._input_dir), file))
                image_width, image_height = image.size

                if image_width > max_width:
                    max_width = image_width

                if image_height > max_height:
                    max_height = image_height

        self._image_size = max(max_width, max_height)

        for file in os.listdir(os.path.join(os.path.dirname(__file__), self._input_dir)):
            if file.endswith(".png"):
                file_path = os.path.join(os.path.join(os.path.dirname(__file__), self._input_dir), file)
                self._resize_image(path=file_path)
                self._scale_image(path=file_path)
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

    def move_generated_images(self):
        """Move the images in generated_images to the train and validation directories."""
        print("Moving the generated images to train and validation...")
        group_counts = defaultdict(int)

        for file in os.listdir(os.path.join(os.path.dirname(__file__), "generated_images")):
            if file.endswith(".png"):
                group = '_'.join(file.split('_', 2)[:2])
                group_counts[group] += 1

                if group_counts[group] <= self._num_training_images:
                    shutil.move(os.path.join(os.path.join(os.path.dirname(__file__), "generated_images"), file),
                                os.path.join(os.path.join(os.path.dirname(__file__), "train"), group))
                else:
                    shutil.move(os.path.join(os.path.join(os.path.dirname(__file__), "generated_images"), file),
                                os.path.join(os.path.join(os.path.dirname(__file__), "validation"), group))

        shutil.rmtree(os.path.join(os.path.dirname(__file__), "generated_images"))

    def train_model(self):
        """Train the model."""
        print("Training the model...")
        data_generator = ImageDataGenerator(rescale=1 / 255)

        train_generator = data_generator.flow_from_directory(directory=os.path.join(os.path.dirname(__file__), "train"),
                                                             target_size=(self._image_size, self._image_size),
                                                             color_mode="rgb",
                                                             batch_size=self._batch_size,
                                                             class_mode="categorical")

        validation_generator = data_generator.flow_from_directory(directory=os.path.join(os.path.dirname(__file__),
                                                                                         "validation"),
                                                                  target_size=(self._image_size, self._image_size),
                                                                  color_mode="rgb",
                                                                  batch_size=self._batch_size,
                                                                  class_mode="categorical")

        self._model.add(Conv2D(filters=32,
                               kernel_size=(3, 3),
                               activation="relu",
                               input_shape=(self._image_size, self._image_size, self._num_channels)))
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
    bc = BhatkhandeClassifier(num_images_per_sample=1000,
                              percentage_train=0.8,
                              samples_directory="samples")
    bc.generate_images()
    bc.move_generated_images()
    bc.train_model()
    bc.save_model()


if __name__ == "__main__":
    main()
