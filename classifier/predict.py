from PIL import Image
from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.preprocessing.image import img_to_array, load_img
import os


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


def make_thumbnail(path):
    """Make a thumbnail for the image at the given path."""
    image = Image.open(path)
    dimensions = (73, 73)
    image.thumbnail(dimensions, Image.ANTIALIAS)
    image.save(path)


def pad_image(path):
    """Pad the image at the given path."""
    image = Image.open(path)
    image_width, image_height = image.size
    maximum_width, maximum_height = 73, 73

    if image_width < maximum_width or image_height < maximum_height:
        background = Image.new("RGBA", (maximum_width, maximum_height), (255, 255, 255, 255))
        offset = round((maximum_width - image_width) / 2), round((maximum_height - image_height) / 2)
        background.paste(image, offset)
        background.save(path)


def scale_image(path):
    """Scale the image at the given path."""
    image = Image.open(path)
    dimensions = (128, 128)
    resized_image = image.resize(dimensions)
    resized_image.save(path)


def reshape_image(path):
    """Return the reshaped image at the given path."""
    image = img_to_array(load_img(path)) / 255
    return image.reshape((1,) + image.shape)


def prepare_image(image_path):
    """Return the image at image_path with the correct dimensions."""
    make_thumbnail(path=image_path)
    pad_image(path=image_path)
    scale_image(path=image_path)
    return reshape_image(path=image_path)


def main():
    """Main function."""
    model = load_model()

    if model is not None:
        class_labels = ['flat_seven', 'flat_six', 'flat_three', 'flat_two', 'sharp_four']
        # TODO: modify the path to the folder containing the images, if necessary
        folder = os.path.join(os.path.dirname(__file__), "corpus")

        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    prediction = model.predict_proba(prepare_image(image_path=os.path.join(folder, file)))
                    label = class_labels[prediction.argmax(axis=-1)[0]]
                    probability = max(prediction[0])
                    print('Prediction label:', label)
                    print('Probability:', probability)
        else:
            print("The folder could not be found.")
    else:
        print("The model could not be loaded.")


if __name__ == "__main__":
    main()
