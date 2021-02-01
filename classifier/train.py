"""Train class."""
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.callbacks import History
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

from classifier.constants import TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, VALID_CLASS_NAMES
from classifier.path_utils import is_file, is_dir, has_valid_image_file_extension, get_entries, \
    get_path, get_parent_dir, get_path_as_str


class Train:
    """Train class."""
    def __init__(self, data_dir: str):
        """
        Train constructor.
        :param data_dir: a str
        """
        self.parent_dir_path = get_parent_dir(get_path(__file__))
        data_dir_path = get_path(data_dir)
        self.train_dir_path = data_dir_path / 'train'
        self.val_dir_path = data_dir_path / 'val'
        self.num_classes = self.validate_data_dir()
        self.model = self.get_model()

    def validate_data_dir(self) -> int:
        """
        Validate the data directory.
        :return: an int (the number of classes)
        """
        self.validate_data_dirs()
        train_classes = self.validate_data_sub_dir(self.train_dir_path)
        val_classes = self.validate_data_sub_dir(self.val_dir_path)

        if train_classes != val_classes:
            raise Exception('The train and val directories must contain the same classes')

        return len(train_classes)

    def validate_data_dirs(self):
        """
        Validate the data directories.
        :return: nothing
        """
        if not is_dir(self.train_dir_path):
            raise Exception('The data directory must contain a train directory')

        if not is_dir(self.val_dir_path):
            raise Exception('The data directory must contain a val directory')

    @staticmethod
    def validate_data_sub_dir(sub_dir: Path) -> list:
        """
        Validate the given data directory subdirectory.
        :param sub_dir: a Path
        :return: a list (the subdirectory class names)
        """
        class_names = []

        for sub_dir_entry in get_entries(sub_dir):
            if not is_dir(sub_dir_entry):
                raise Exception('Each data subdirectory must only contain folders')

            class_name = sub_dir_entry.name

            if class_name not in VALID_CLASS_NAMES:
                raise Exception('Each data subdirectory must have a valid class name')

            for sub_dir_sub_entry in get_entries(sub_dir_entry):
                if not is_file(sub_dir_sub_entry):
                    raise Exception('Each data subdirectory folder must only contain files')

                if not has_valid_image_file_extension(sub_dir_sub_entry):
                    raise Exception('Each data subdirectory folder must only contain PNG images')

            class_names.append(class_name)

        return class_names

    def get_model(self) -> Sequential:
        """
        Get the model.
        :return: a Sequential object
        """
        # sample model to be replaced
        model = Sequential()
        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3)))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(units=self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, num_epochs: int, batch_size: int, save_model: bool, save_history: bool):
        """
        Train the model.
        :param num_epochs: an int
        :param batch_size: an int
        :param save_model: a bool
        :param save_history: a bool
        :return: nothing
        """
        train_generator = ImageDataGenerator(rotation_range=1,
                                             width_shift_range=0.02,
                                             height_shift_range=0.02,
                                             rescale=1 / 255,
                                             shear_range=0.05,
                                             zoom_range=0.05,
                                             brightness_range=(0, 2),
                                             fill_mode='nearest')
        val_generator = ImageDataGenerator(rescale=1 / 255)
        target_size = (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH)
        class_mode = 'categorical'
        train_data = train_generator.flow_from_directory(self.train_dir_path,
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         class_mode=class_mode)
        val_data = val_generator.flow_from_directory(self.val_dir_path,
                                                     target_size=target_size,
                                                     batch_size=batch_size,
                                                     class_mode=class_mode)
        history = self.model.fit(train_data,
                                 steps_per_epoch=train_data.samples // batch_size,
                                 validation_data=val_data,
                                 validation_steps=val_data.samples // batch_size,
                                 epochs=num_epochs)

        if save_model:
            self.save_model()
            self.save_index_to_class_mapping(train_data.class_indices)

        if save_history:
            self.save_history(history)

    def save_model(self):
        """
        Save the model.
        :return: nothing
        """
        path = self.parent_dir_path / 'my_model'
        self.model.save(path)

    def save_index_to_class_mapping(self, mapping: dict):
        """
        Save the index-to-class mapping.
        :param mapping: a dict
        :return: nothing
        """
        # invert the mapping so that the indices are the dictionary keys
        data = {value: key for key, value in mapping.items()}
        path = self.parent_dir_path / 'mapping.json'

        with path.open('w') as file:
            json.dump(data, file)

    def save_history(self, history: History):
        """
        Save the history.
        :param history: a History object
        :return: nothing
        """
        epochs = history.epoch
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.suptitle('Train History')
        ax1.plot(epochs, train_loss)
        ax1.plot(epochs, val_loss)
        ax1.set_ylabel('Loss')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.legend(['Train', 'Val'], loc='upper right', title='Data:')
        ax2.plot(epochs, train_accuracy)
        ax2.plot(epochs, val_accuracy)
        ax2.set_ylabel('Accuracy')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.legend(['Train', 'Val'], loc='lower right', title='Data:')
        plt.xlabel('Epoch')
        path = self.parent_dir_path / 'model_history.png'
        path_as_str = get_path_as_str(path)
        plt.savefig(path_as_str)


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('num_epochs', type=int)
@click.argument('batch_size', type=int)
@click.option('--save_model', '-sm', type=bool, is_flag=True)
@click.option('--save_history', '-sh', type=bool, is_flag=True)
def main(data_dir: str, num_epochs: int, batch_size: int, save_model: bool, save_history: bool):
    """
    The main function.
    :param data_dir: a str
    :param num_epochs: an int
    :param batch_size: an int
    :param save_model: a bool
    :param save_history: a bool
    :return: nothing
    """
    try:
        print('Starting training')
        train = Train(data_dir)
        train.train_model(num_epochs, batch_size, save_model, save_history)
        print('Successfully completed training')
    except Exception as error:  # pylint: disable=broad-except
        print('Encountered exception:', error)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
