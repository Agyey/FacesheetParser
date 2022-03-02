import cv2
import numpy as np
import os
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import traceback
import easyocr
from ocr import box_extraction
import re
from align_images import align_images

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

IMAGE_DIRECTORY = "facesheets"
CONFIG_DIRECTORY = "bin"
DEBUG = 0
TEMPLATE_IMAGE = cv2.imread("./tests/template_1.jpg")
junk = ['©', ',', ':', ';', '.']

print("Loading Model...")
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
print("Model Loaded")


def draw_box(image, x, y, w, h, colour_space=(0, 0, 255), border_size=2):
    """
    Draws a box around an object on an image
    :param image: image to draw the box
    :param x: top left corner x-coordinate
    :param y: top left corner y-coordinate
    :param w: width of the bounding box
    :param h: height of the bounding box
    :param colour_space: a tuple defining the BGR values for the
     bounding box border
    :param border_size: Size of bounding box border
    """
    cv2.rectangle(image, (x, y), (x + w, y + h), colour_space, border_size)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes the image based on the width or height, keeping aspect ratio
    constant
    :param image: Image to be resized
    :param width: Width of the final image
    :param height: Height of the final image
    :param inter: Interpolation method (default method, cv2.INTER_AREA)
    :return: resized image
    """
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


def show_image(image, window_name="untitled"):
    """
    Displays a given image in a named window (default name, 'Untitled')
    :param image: Image to be displayed
    :param window_name: Name of the window (default name, 'Untitled')
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def name_parser(text):
    """
    Extracts first, middle, last name from a text
    :param text: Text with name
    :return: first, middle, last name
    """
    first_name, middle_name, last_name = '', '', ''
    if "," in text:
        text = text
        last_name, remaining_part = text.split(",")
        name_parts = [x.strip() for x in remaining_part.strip().split(" ") if x.strip()]
        if len(name_parts) == 2:
            first_name, middle_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
        else:
            first_name = last_name
            last_name = None
    else:
        name_parts = text.strip().split(" ")
        if len(name_parts) == 3:
            first_name, middle_name, last_name = name_parts
        elif len(name_parts) == 2:
            first_name, last_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
    return first_name.strip(), middle_name.strip(), last_name.strip()


def extract_field(label, field_width, data):
    """
    Extracts field from an image given the field label and width
    :param label: Field label to be extracted, as it appears on the image
    :param field_width: Distance of the value from the image
    :param data: pytesseract data for the image
    :return: extracted field value
    """
    num_points = len(data["level"])
    target_box = None
    field_label = None
    for point in range(num_points):
        text = data["text"][point]
        x, y, w, h = data["left"][point], data["top"][point], \
                     data["width"][point], data["height"][point]
        if text.lower().startswith(label.lower()):
            target_box = (x, y, w, h)
            field_label = text
            break
    candidate_boxes = []
    if target_box:
        for point in range(num_points):
            if float(data["conf"][point]) < 10:
                continue
            text = data["text"][point]
            x, y, w, h = data["left"][point], data["top"][point], \
                         data["width"][point], data["height"][point]
            if abs(y - target_box[1]) < 10 and text != field_label \
                    and ":" not in text and 0 < x - target_box[0] < field_width:
                candidate_boxes.append((text, (x, y, w, h)))
    candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1][0])
    field_value = " ".join(x[0] for x in candidate_boxes)
    return field_value


def number_parser(text):
    number = text.strip()
    number = number.replace("o", "0")
    number = number.replace("O", "0")
    number = number.replace("l", "1")
    return number


def main():
    df = pd.DataFrame()
    for file_name in os.listdir('facesheets'):
        # Skip files
        if '7' not in file_name:
            continue
        images = convert_from_path(f'facesheets/{file_name}')
        c = 0
        fields = {
            (2, 0): "Name", (3, 0): "Address Line 1", (3, 1): "Gender",
            (4, 0): "Address Line 2", (4, 1): "Phone",
            (4, 2): "MRN", (5, 0): "City", (5, 1): "State",
            (5, 2): "Zip", (5, 3): "County", (6, 0): "Sex", (6, 1): "DOB",
            (6, 2): "Age", (8, 0): "Admit Doctor", (17, 1): "Policy Number",
            (18, 0): "Insurance Address"
        }
        entry = dict()
        for image in images:
            # Skip images
            c += 1
            if c in [10]:
                continue
            image = np.array(image)
            aligned_image = align_images(image, TEMPLATE_IMAGE)
            print("Extracting image segments...")
            box_images = box_extraction(aligned_image)
            print("Done.")
            # show_image(aligned_image)
            for i, row in enumerate(box_images):
                # if all([x[0] != i for x in fields]):
                #     continue
                # print(f"Row {i}")
                for j, box_image in enumerate(row):
                    # if all([x != (i, j) for x in fields]):
                    #     continue
                    # print(f"Column {j}")
                    # box_image = np.array(box_image)
                    cv2.imwrite("temp.jpg", box_image)
                    # print(result)
                    if (i, j) in fields:
                        result = reader.readtext('temp.jpg')
                        avg_conf = np.mean([x[-1] for x in result])
                        result = [x[-2] for x in result]
                        print(f"easyocr: {result}")
                        # print(avg_conf)
                        text = pytesseract.image_to_string(box_image).strip()
                        print(f"pytesseract: {text}")
                        # (2, 0): "Name", (3, 0): "Address Line 1", (
                        # 3, 1): "Gender",
                        # (4, 0): "Address Line 2", (4, 1): "Phone",
                        # (4, 2): "MRN", (5, 0): "City", (5, 1): "State",
                        # (5, 2): "Zip", (5, 3): "County", (6, 0): "Sex", (
                        # 6, 1): "DOB",
                        # (6, 2): "Age", (8, 0): "Admit Doctor",
                        # (17, 1): "Policy Number",
                        # (18, 0): "Insurance Address"
                        text = [x for x in re.split("\n+", text) if x][1:]
                        if (i, j) == (2, 0):
                            entry[fields[(i, j)]] = " ".join(re.split("[, ]", text[0])[:3])
                        elif (i, j) == (8, 0):
                            entry[fields[(i, j)]] = " ".join(re.split("[, ]", text[0])[2:])
                        elif (i, j) in [(3, 1), (6, 0)]:
                            entry[fields[(i, j)]] = result[-1] if len(result) > 1 else "NA"
                        elif (i, j) in [(3, 0), (4, 0)]:
                            if text:
                                entry[fields[(i, j)]] = " ".join([x for x in text[0].split(" ") if x not in junk])
                            else:
                                entry[fields[(i, j)]] = "NA"
                        elif (i, j) == (4, 1):
                            entry[fields[(i, j)]] = " ".join(result[-1:])
                        elif (i, j) == (5, 0):
                            entry[fields[(i, j)]] = [x for x in result if x.isupper()][0]
                        elif (i, j) in [(5, 1), (17, 1)]:
                            entry[fields[(i, j)]] = result[-1]
                        elif (i, j) == (6, 2):
                            entry[fields[(i, j)]] = " ".join(result[1:])
                        elif (i, j) == (5, 2):
                            entry[fields[(i, j)]] = result[-1]
                        else:
                            if text:
                                entry[fields[(i, j)]] = text[0]
                            else:
                                entry[fields[(i, j)]] = "NA"
                        print(f"{fields[(i, j)]}: {entry[fields[(i, j)]]}")
                            # entry[fields[(i, j)] + "_easyocr"] = " ".join([x[-2] for x in result])
                        # entry[fields[(i, j)] + "_tesseract"] = text
                    # show_image(box_image)
            # image.save("temp.jpg")
            df = df.append(entry, ignore_index=True)
            df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
