import unittest
from classifier import predict
from PIL import Image
import os
import shutil


class TestMainFunctions(unittest.TestCase):
    def test_make_thumbnail(self):
        """Test make_thumbnail."""
        os.chdir(os.path.dirname(__file__))
        predict.make_thumbnail("test_images/sample.png")
        image = Image.open("test_images/sample.png")
        image_width, image_height = image.size
        maximum_width, maximum_height = 73, 73
        self.assertTrue(image_width <= maximum_width)
        self.assertTrue(image_height <= maximum_height)
        os.remove("test_images/sample.png")
        shutil.copy("test_images/original.png", "test_images/sample.png")

    def test_resize_image(self):
        """Test resize_image."""
        os.chdir(os.path.dirname(__file__))
        predict.make_thumbnail("test_images/sample.png")
        predict.pad_image("test_images/sample.png")
        image = Image.open("test_images/sample.png")
        image_width, image_height = image.size
        correct_width, correct_height = 73, 73
        self.assertTrue(image_width == correct_width)
        self.assertTrue(image_height == correct_height)
        os.remove("test_images/sample.png")
        shutil.copy("test_images/original.png", "test_images/sample.png")

    def test_pad_image(self):
        """Test pad_image."""
        os.chdir(os.path.dirname(__file__))
        predict.make_thumbnail("test_images/sample.png")
        predict.pad_image("test_images/sample.png")
        predict.scale_image("test_images/sample.png")
        image = Image.open("test_images/sample.png")
        image_width, image_height = image.size
        correct_width, correct_height = 128, 128
        self.assertTrue(image_width == correct_width)
        self.assertTrue(image_height == correct_height)
        os.remove("test_images/sample.png")
        shutil.copy("test_images/original.png", "test_images/sample.png")


if __name__ == "__main__":
    unittest.main()
