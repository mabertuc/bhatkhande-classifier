import os
import cv2
import numpy as np
from predict import *


def is_valid_contour(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    return w > 10 and h > 10 and h < 150


def sort_y(cnt):
    _, y, _, h = cv2.boundingRect(cnt[0])
    return (2*y + h)/2


def sort_x(cnt):
    # TODO use average of x val
    x, _, w, _ = cv2.boundingRect(cnt[0])
    return (2*x + w)/2


def crop_image(image, cnt):
    """ Crop an image to the bounding box of the given contour """
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]


def draw_result(image, contour, label, probability):
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = 3*x, 3*y, 3*w, 3*h
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (x, y),
                font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def get_class(model, page_img, contour):
    class_labels = get_class_labels()
    """ Return the class label and probability given the model, image of the page, and the contour on the page """
    cropped_char = crop_image(page_img, contour)
    cv2.imwrite("tmp.png", cropped_char)
    image = prepare_image(image_path="tmp.png")
    prediction = model.predict_proba(image)
    label = class_labels[prediction.argmax(axis=-1)[0]]
    probability = max(prediction[0])
    print("*" * 40)
    print('Prediction label:', label)
    print('Probability:', probability)
    return (label, probability)


def detect_contours(composition_image):
    processed = cv2.cvtColor(composition_image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((4, 4), np.uint8)
    processed = cv2.erode(processed, kernel, iterations=1)

    _, processed = cv2.threshold(processed, 192, 255, cv2.THRESH_BINARY)

    processed = cv2.bilateralFilter(processed, 4, 10, 10)

    processed = cv2.Canny(processed, 800, 800)

    contours, hierarchy = cv2.findContours(
        processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ordered_contours = []
    row = []
    rows = []
    contours.sort(key=sort_y)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not is_valid_contour(c, composition_image):
            continue

        if (len(row) == 0):
            row.append(c)

        else:
            x_last, y_last, w_last, h_last = cv2.boundingRect(
                row[len(row)-1][0])
            if abs(y-y_last) < 20:
                # same row
                row.append(c)
            else:
                # new row
                row.sort(key=sort_x)
                rows.append(row)
                row = [c]
    rows.append(row)
    ordered_contours = [contour for row in rows for contour in row]
    filtered_contours = []
    for i, c in enumerate(ordered_contours):
        x, y, w, h = cv2.boundingRect(c)
        if i > 0:
            x_last, y_last, w_last, h_last = cv2.boundingRect(ordered_contours[i-1])
            if abs(x-x_last) > 6:
                filtered_contours.append(c)
        else:
            filtered_contours.append(c)
    return filtered_contours


model = load_model()


if model is None:
    print("The model could not be loaded.")
    exit(0)

composition_folder = os.path.join(os.path.dirname(__file__), "compositions")
output_folder = os.path.join(os.path.dirname(__file__), "output")
valid_file_extension = ".png"

if os.path.isdir(composition_folder):
    for file in os.listdir(composition_folder):
        if file.endswith(valid_file_extension):
            print("Processing", file)
            predictions = []
            page_img = cv2.imread(os.path.join(composition_folder, file))
            
            draw_img = cv2.resize(page_img, None, fx=3, fy=3)

            for i, contour in enumerate(detect_contours(page_img)):
                label, probability = get_class(model, page_img, contour)
                predictions.append(label)
                draw_result(draw_img, contour, label, probability)

            output_file_base = os.path.join(output_folder, file.split(".")[0])

            cv2.imwrite(output_file_base + "_processed.png", draw_img)
            stream = generate_score(predictions)
            stream.write('midi', fp=output_file_base + '.mid')
            stream.write('xml', fp=output_file_base + '.xml')



else:
    print("The folder could not be loaded")
    exit(0)
