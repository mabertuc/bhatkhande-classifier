import os
import classifier.predict
import classifier.composition_parser


class CompositionAnalyzer:
    """
    Attributes:
        _valid_file_extension: a string indicating the valid file extension for images
        _composition_directory: the directory containing the compositions to analyze
        _output_directory: the directory containing the analyzed compositions
        _predict: an instance of the Predict class
        _composition_parser: an instance of the CompositionParser class
    Methods:
        _check_directory_status: checks if both the composition and output directories exist
        analyze_compositions: analyzes the compositions in the compositions directory
    """
    def __init__(self, composition_directory, output_directory):
        """
        CompositionAnalyzer constructor.
        :param composition_directory: a string
        :param output_directory: a string
        """
        self._valid_file_extension = ".png"
        self._composition_directory = composition_directory
        self._output_directory = output_directory
        self._check_directory_status()
        self._predict = classifier.predict.Predict()
        self._composition_parser = classifier.composition_parser.CompositionParser()

    def _check_directory_status(self):
        """
        Check the directory status.
        :return: nothing
        """
        if not os.path.isdir(self._composition_directory) or not os.path.isdir(self._output_directory):
            raise Exception("The composition directory or the output directory do not exist")

    def analyze_compositions(self):
        """
        Analyze compositions.
        :return: nothing
        """
        for file in os.listdir(self._composition_directory):
            if file.endswith(self._valid_file_extension):
                path = os.path.join(self._composition_directory, file)
                page_image = self._composition_parser.load_page_image(path=path)
                draw_image = self._composition_parser.load_resized_page_image(src=page_image, dsize=None, fx=3, fy=3)

                for i, contour in enumerate(self._composition_parser.detect_contours(page_image)):
                    cropped_char = self._composition_parser.crop_image(page_image, contour)
                    self._composition_parser.write_image("tmp.png", cropped_char)
                    prediction = self._predict.make_prediction("tmp.png")
                    prediction_label = prediction[0]
                    self._composition_parser.draw_result(draw_image, contour, prediction_label)

                output_file_base = os.path.join(self._output_directory, file.split(".")[0])
                self._composition_parser.write_image(output_file_base + "_processed.png", draw_image)
                self._predict.generate_stream()
                self._predict.save_stream(file_type="musicxml", path=output_file_base + ".xml")
                self._predict.reset()


def main():
    """
    Main function.
    :return: nothing
    """
    composition_directory = os.path.join(os.path.dirname(__file__), "compositions")
    output_directory = os.path.join(os.path.dirname(__file__), "output")

    try:
        composition_analyzer = CompositionAnalyzer(composition_directory=composition_directory,
                                                   output_directory=output_directory)
        composition_analyzer.analyze_compositions()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
