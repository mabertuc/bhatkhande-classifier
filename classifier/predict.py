import classifier.image_util
from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import os
import music21


class Predict:
    """
    Attributes:
        ...
    Methods:
        ...
    """
    def __init__(self):
        """Predict constructor."""
        self._valid_file_extension = ".png"
        self._scale_degree_to_note = {}
        self._set_scale_degree_to_note()
        self._class_labels = []
        self._set_class_labels()
        self._model = self._load_model()
        self._predictions = []
        self._stream = music21.stream.Stream()

    def _set_scale_degree_to_note(self):
        """
        Set _scale_degree_to_note.
        :return: nothing
        """
        self._scale_degree_to_note = {
            "flat-seven-low": "B-4",
            "seven-low": "B4",
            "one": "C4",
            "flat-two": "D-4",
            "two": "D4",
            "flat-three": "E-4",
            "three": "E4",
            "four": "F4",
            "sharp-four": "F#4",
            "five": "G4",
            "flat-six": "A-4",
            "six": "A4",
            "flat-seven": "B-4",
            "seven": "B4",
            "one-high": "C5",
            "flat-two-high": "D-5",
            "two-high": "D5",
            "three-high": "E5",
            "five-high": "G5",
        }

    def _set_class_labels(self):
        """
        Set _class_labels.
        :return: nothing
        """
        directory = os.path.join(os.path.dirname(__file__), "new_samples")

        for folder in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, folder)):
                class_label = folder
                self._class_labels.append(class_label)

        self._class_labels = sorted(self._class_labels)

    @staticmethod
    def _load_model():
        """
        Load the model.
        :return: a Keras model if one is found, None otherwise
        """
        directory = os.path.dirname(__file__)
        model_json_path = os.path.join(directory, "model.json")
        model_h5_path = os.path.join(directory, "model.h5")

        if os.path.isfile(model_json_path) and os.path.isfile(model_h5_path):
            with open(model_json_path, 'r') as file:
                model = model_from_json(file.read())

            model.load_weights(model_h5_path)
            model.compile(loss=categorical_crossentropy,
                          optimizer=Adadelta(),
                          metrics=["accuracy"])
            return model
        else:
            raise Exception("The model could not be loaded")

    @staticmethod
    def _prepare_image(path):
        """
        Prepare the image at the given path.
        :param path: a string
        :return: a 3D numpy array
        """
        classifier.image_util.pad_image(path=path)
        classifier.image_util.resize_image(path=path)
        image_array = classifier.image_util.reshape_image(path=path)
        return image_array

    def make_prediction(self, path):
        """
        Make a prediction.
        :param path: a string
        :return: a tuple of int
        """
        if self._model is not None and path.endswith(self._valid_file_extension):
            image = self._prepare_image(path=path)
            prediction = self._model.predict_proba(image)
            label = self._class_labels[prediction.argmax(axis=-1)[0]]
            probability = max(prediction[0])
            self._predictions.append((path, label, probability))
            return label, probability

    def _get_note(self, scale_degree):
        """
        Get the note.
        :param scale_degree: a string
        :return: a string
        """
        return self._scale_degree_to_note[scale_degree]

    def generate_stream(self):
        """
        Generate the stream.
        :return: nothing
        """
        duration = "quarter"
        previous_note = None

        for prediction in self._predictions:
            label = prediction[1]

            if label != "divider":
                if "rhythm" in label:
                    continue
                elif label == "continue":
                    note = previous_note
                    self._stream.append(music21.note.Note(note, type=duration))
                    previous_note = note
                elif label == "rest":
                    self._stream.append(music21.note.Rest())
                    previous_note = None
                else:
                    note = self._get_note(label)
                    self._stream.append(music21.note.Note(note, type=duration))
                    previous_note = note

    def get_stream(self):
        """
        Get the stream.
        :return: a music21.stream.Stream
        """
        return self._stream

    def save_stream(self, file_type, path):
        """
        Save the stream.
        :param file_type: a string
        :param path: a string
        :return: nothing
        """
        self._stream.write(file_type, path)

    def reset(self):
        """
        Reset class attributes.
        :return: nothing
        """
        self._predictions = []
        self._stream = music21.stream.Stream()


def main():
    """Main function."""
    try:
        predict = Predict()
        directory = os.path.join(os.path.dirname(__file__), "corpus")

        for file in os.listdir(directory):
            predict.make_prediction(path=os.path.join(directory, file))

        predict.generate_stream()
        predict.save_stream(file_type="musicxml", path=os.path.join(os.path.dirname(__file__), "score.xml"))
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
