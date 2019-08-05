from PIL import Image
from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.preprocessing.image import img_to_array, load_img
import os
import music21


def load_model():
    """Load the model."""
    if os.path.isfile(os.path.join(os.path.dirname(__file__), "model.json")) and \
            os.path.isfile(os.path.join(os.path.dirname(__file__), "model.h5")):
        with open(os.path.join(os.path.dirname(__file__), "model.json"), 'r') as file:
            model = model_from_json(file.read())
        model.load_weights(os.path.join(os.path.dirname(__file__), "model.h5"))
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=["accuracy"])
        return model
    return None


def pad_image(path):
    """Pad the image."""
    image = Image.open(path)
    current_image_width, current_image_height = image.size
    target_image_side_length = max(current_image_height, current_image_width)
    background = get_background(target_image_side_length=target_image_side_length)
    background_offset = get_background_offset(target_image_side_length=target_image_side_length,
                                              current_image_width=current_image_width,
                                              current_image_height=current_image_height)
    background.paste(image, background_offset)
    background.save(path)


def get_background(target_image_side_length):
    """Get the background."""
    background_color_mode = "RGBA"
    background_size = (target_image_side_length, target_image_side_length)
    background_color = (255, 255, 255, 255)
    background = Image.new(background_color_mode, background_size, background_color)
    return background


def get_background_offset(target_image_side_length, current_image_width, current_image_height):
    """Get the background offset."""
    x_coord = round((target_image_side_length - current_image_width) / 2)
    y_coord = round((target_image_side_length - current_image_height) / 2)
    return x_coord, y_coord


def resize_image(path):
    """Resize the image at the given path."""
    image = Image.open(path)
    dimensions = (100, 100)
    resized_image = image.resize(dimensions)
    resized_image.save(path)


def reshape_image(path):
    """Return the reshaped image at the given path."""
    image = img_to_array(load_img(path)) / 255
    return image.reshape((1,) + image.shape)


def prepare_image(image_path):
    """Return an image array of the image at image_path."""
    pad_image(path=image_path)
    resize_image(path=image_path)
    image_array = reshape_image(path=image_path)
    return image_array


def get_class_labels():
    """Get the class labels."""
    directory = os.path.join(os.path.dirname(__file__), "samples")
    valid_file_extension = ".png"
    class_labels = []

    for file in os.listdir(directory):
        if file.endswith(valid_file_extension):
            class_label = file.split('.')[0]
            class_labels.append(class_label)

    return sorted(class_labels)


def get_note(scale_degree):
    return {
        "one": "C",
        "flat-two": "D-",
        "two": "D",
        "flat-three": "E-",
        "three": "E",
        "four": "F",
        "sharp-four": "F#",
        "five": "G",
        "flat-six": "A-",
        "six": "A",
        "flat-seven": "B-",
        "seven": "B",
    }[scale_degree]


def generate_score(predictions):
    """Generate score."""
    stream = music21.stream.Stream()
    duration = "quarter"
    previous_note = None

    for prediction in predictions:
        if prediction != "divider":
            if prediction == "continue":
                note = previous_note
                stream.append(music21.note.Note(note, type=duration))
                previous_note = note
            elif prediction == "rest":
                stream.append(music21.note.Rest())
                previous_note = None
            else:
                note = get_note(prediction)
                stream.append(music21.note.Note(note, type=duration))
                previous_note = note

    stream.show()


def main():
    """Main function."""
    model = load_model()

    if model is not None:
        class_labels = get_class_labels()
        corpus_folder = os.path.join(os.path.dirname(__file__), "corpus")
        valid_file_extension = ".png"
        predictions = []

        if os.path.isdir(corpus_folder):
            for file in os.listdir(corpus_folder):
                if file.endswith(valid_file_extension):
                    image = prepare_image(image_path=os.path.join(corpus_folder, file))
                    prediction = model.predict_proba(image)
                    label = class_labels[prediction.argmax(axis=-1)[0]]
                    predictions.append(label)
                    probability = max(prediction[0])
                    print("*" * 40)
                    print('File:', file)
                    print('Prediction label:', label)
                    print('Probability:', probability)

            generate_score(predictions)
        else:
            print("The folder could not be found.")
    else:
        print("The model could not be loaded.")


if __name__ == "__main__":
    main()
