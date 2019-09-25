import cv2
import numpy as np


class CompositionParser:
    """
    Attributes:
        ...
    Methods:
        ...
    """
    @staticmethod
    def load_page_image(path):
        """
        Load the page image at the given path.
        :param path: a string
        :return: an image
        """
        return cv2.imread(path)

    @staticmethod
    def load_resized_page_image(src, dsize, fx, fy):
        return cv2.resize(src=src, dsize=dsize, fx=fx, fy=fy)

    @staticmethod
    def write_image(name, image):
        cv2.imwrite(filename=name, img=image)

    @staticmethod
    def is_valid_contour(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return w > 10 and 10 < h < 150

    @staticmethod
    def _sort_y(cnt):
        _, y, _, h = cv2.boundingRect(cnt[0])
        return (2 * y + h) / 2

    @staticmethod
    def _sort_x(cnt):
        x, _, w, _ = cv2.boundingRect(cnt[0])
        return (2 * x + w) / 2

    def detect_contours(self, composition_image):
        processed = cv2.cvtColor(composition_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((4, 4), np.uint8)
        processed = cv2.erode(processed, kernel, iterations=1)
        _, processed = cv2.threshold(processed, 192, 255, cv2.THRESH_BINARY)
        processed = cv2.bilateralFilter(processed, 4, 10, 10)
        processed = cv2.Canny(processed, 800, 800)
        contours, hierarchy = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        row = []
        rows = []
        contours.sort(key=self._sort_y)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if not self.is_valid_contour(contour):
                continue

            if len(row) == 0:
                row.append(contour)
            else:
                x_last, y_last, w_last, h_last = cv2.boundingRect(row[len(row) - 1][0])

                if abs(y - y_last) < 20:
                    row.append(contour)
                else:
                    row.sort(key=self._sort_x)
                    rows.append(row)
                    row = [contour]

        rows.append(row)
        ordered_contours = [contour for row in rows for contour in row]
        filtered_contours = []

        for i, contour in enumerate(ordered_contours):
            x, y, w, h = cv2.boundingRect(contour)

            if i > 0:
                x_last, y_last, w_last, h_last = cv2.boundingRect(ordered_contours[i - 1])

                if abs(x-x_last) > 6:
                    filtered_contours.append(contour)
            else:
                filtered_contours.append(contour)

        return filtered_contours

    @staticmethod
    def crop_image(image, cnt):
        """ Crop an image to the bounding box of the given contour """
        x, y, w, h = cv2.boundingRect(cnt)
        return image[y:y+h, x:x+w]

    @staticmethod
    def draw_result(image, contour, label):
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = 3*x, 3*y, 3*w, 3*h
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (x, y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
