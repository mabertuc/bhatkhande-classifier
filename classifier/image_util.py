from PIL import Image
from keras.preprocessing.image import img_to_array, load_img


def pad_image(path):
    """
    Pad the image.
    :param path: a string
    :return: nothing
    """
    image = Image.open(path)
    current_image_width, current_image_height = image.size
    side_length = max(current_image_height, current_image_width)
    background = get_background(side_length=side_length)
    background_offset = get_background_offset(side_length=side_length,
                                              current_image_width=current_image_width,
                                              current_image_height=current_image_height)
    background.paste(image, background_offset)
    background.save(path)


def get_background(side_length):
    """
    Get the background.
    :param side_length: an int
    :return: an image
    """
    background_color_mode = "RGBA"
    background_size = (side_length, side_length)
    background_color = (255, 255, 255, 255)
    background = Image.new(background_color_mode, background_size, background_color)
    return background


def get_background_offset(side_length, current_image_width, current_image_height):
    """
    Get the background offset.
    :param side_length: an int
    :param current_image_width: an int
    :param current_image_height: an int
    :return: a tuple of floating point numbers
    """
    x_coord = round((side_length - current_image_width) / 2)
    y_coord = round((side_length - current_image_height) / 2)
    return x_coord, y_coord


def resize_image(path):
    """
    Resize the image.
    :param path: a string
    :return: nothing
    """
    image = Image.open(path)
    dimensions = (30, 30)
    resized_image = image.resize(dimensions)
    resized_image.save(path)


def reshape_image(path):
    """
    Reshape the image.
    :param path: a string
    :return: a 3D numpy array
    """
    image = img_to_array(load_img(path)) / 255
    return image.reshape((1,) + image.shape)
