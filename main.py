import os

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
# pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract.exe'

IMAGE_DIRECTORY = "facesheets"
CONFIG_DIRECTORY = "bin"
DEBUG = 0


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
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
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
    # cv2.destroyAllWindows()


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
        name_parts = remaining_part.strip().split(" ")
        if len(name_parts) == 2:
            first_name, middle_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
        else:
            first_name = last_name
            last_name = None
    else:
        name_parts = text.split(" ")
        if len(name_parts) == 3:
            first_name, middle_name, last_name = name_parts
        elif len(name_parts) == 2:
            first_name, middle_name = name_parts
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
        x, y, w, h = data["left"][point], data["top"][point],\
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


def get_boxes(image):
    """
    Returns a list of bounding boxes from an image, resolved to their nearest
    neighbours
    :param image: image to extract bounding boxes from
    :return: list of bounding boxes
    """
    data = pytesseract.image_to_data(image,
                                         output_type=pytesseract.Output.DICT)
    num_points = len(data["level"])
    final_data = {"left": [], "width": [], "top": [], "height": []}
    for point in range(num_points):
        x, y, w, h = data["left"][point], data["top"][point], \
                     data["width"][point], data["height"][point]
        for compare_point in range(num_points):
            x_c, y_c, w_c, h_c = data["left"][compare_point], \
                                 data["top"][compare_point], \
                                 data["width"][compare_point], \
                                 data["height"][compare_point]
            if point == compare_point or (y - y_c) > 10:
                continue
    return data


def main():
    for file_name in os.listdir('facesheets'):
        df = pd.DataFrame(
            columns=[
                "First Name", "Middle Name", "Last Name",
                "Visit ID", "MRN", "DOB", "Age", "Gender", "SSN",
                "Admitting MD First Name", "Admitting MD Middle Name", "Admitting MD Last Name",
                "Address Line 1", "Address Line 2", "City", "State", "Zip", "Phone Number", "Number Type",
                "Policy Number", "Insurance Address Line 1", "Insurance City", "Insurance State",  "Insurance Zip"
            ])
        images = convert_from_path(f'facesheets/{file_name}')
        c = 0
        for image in images:
            c += 1
            if c == 4:
                break
            entry = {}
            image.save("temp.jpg")
            image = cv2.imread("temp.jpg")
            data = get_boxes(np.asarray(image))
            num_points = len(data["level"])
            boxes = []
            try:
                for point in range(num_points):
                    x, y, w, h = data["left"][point], data["top"][point], \
                                 data["width"][point], data["height"][point]
                    text = data["text"][point]
                    if text.strip():
                        boxes.append(((x, y, w, h), text))
                compared_boxes = []
                sorted_boxes = sorted(boxes, key=lambda x: x[0][1])
                rows = []
                for i, box in enumerate(sorted_boxes):
                    if box in compared_boxes:
                        continue
                    coords, text = box
                    x, y, w, h = coords
                    row = [box]
                    compared_boxes.append(box)
                    for box_c in sorted_boxes[i+1:]:
                        if box_c in compared_boxes:
                            continue
                        coords, text_c = box_c
                        x_c, y_c, w_c, h_c = coords
                        if abs(y - y_c) < 15 and abs(h - h_c) < 15:
                            row.append(box_c)
                            compared_boxes.append(box_c)
                        else:
                            break
                    rows.append(row)
                    x = min(row, key=lambda k: k[0][0])[0][0]
                    y = min(row, key=lambda k: k[0][1])[0][1]
                    w = max(k[0][0] + k[0][2] for k in row) - x
                    h = max(k[0][1] + k[0][3] for k in row) - y
                    # draw_box(image, x, y, w, h)
                    # resized_image = image_resize(image, width=1000)
                    # show_image(resized_image)
                fields = [
                    "name", "visit", "mr", "birthdate", "age", "gender", "ssn", "admitting md", "patient address/phone",
                    "insurance addr", "city", "state", "zip", "policy"]
                field_mapping = {
                    "visit": "Visit ID", "mr": "MRN", "birthdate": "DOB", "age": "Age", "gender": "Gender", "ssn": "SSN",
                    "policy": "Policy Number", "insurance addr": "Insurance Address Line 1", "city": "Insurance City",
                    "state": "Insurance State", "zip": "Insurance Zip"
                }
                found_fields = []
                for r, row in enumerate(rows):
                    sorted_row = sorted(row, key=lambda x: x[0][0])

                    for i, box in enumerate(sorted_row):
                        coords, text = box
                        x, y, w, h = coords
                        field_name = ""
                        for field in filter(lambda k: k not in found_fields, fields):
                            matching = False
                            for j, value in enumerate(field.split()):
                                if text.lower().startswith(value):
                                    matching = True
                                    if i + j + 1 >= len(sorted_row):
                                        if j != len(field.split()) - 1:
                                            matching = False
                                        break
                                    box = sorted_row[i+j+1]
                                    coords, text = box
                                else:
                                    matching = False
                                    break
                            if matching:
                                field_name = field
                                found_fields.append(field)
                                break
                        if not field_name:
                            continue
                        field_value = ""
                        j = i + len(field_name.split())
                        while True and j < len(sorted_row):
                            box_c = sorted_row[j]
                            coords_c, text_c = box_c
                            if ":" not in text_c:
                                break
                            j += 1
                        x_c, y_c, w_c, h_c = coords_c
                        first_dist = x_c - x - w
                        last_dist = 0
                        last_edge = 0
                        for box_c in sorted_row[j:]:
                            coords_c, text_c = box_c
                            x_c, y_c, w_c, h_c = coords_c
                            if last_edge != 0:
                                if last_dist == 0:
                                    last_dist = x_c - last_edge
                                    if last_dist > 0.5 * first_dist:
                                        break
                                else:
                                    if x_c - last_edge > 2 * last_dist:
                                        break
                            if last_edge != 0:
                                dist = x_c - last_edge
                                if dist > first_dist:
                                    break
                            last_edge = x_c + w_c
                            field_value += " " + text_c
                        if field_name == "patient address/phone":
                            text = ""
                            k = r + 1
                            while len(rows[k]) != 1:
                                k += 1
                            segments = []
                            print(rows[r+1:k])
                            for addr_row in rows[r+1:k]:
                                sorted_addr_row = sorted(addr_row, key=lambda x: x[0][0])
                                compared_boxes = []
                                for l, ele in enumerate(sorted_addr_row):
                                    if ele in compared_boxes:
                                        continue
                                    coords, text = ele
                                    x, y, w, h = coords
                                    for ele_c in sorted_addr_row[l+1:]:
                                        coords_c, text_c = ele_c
                                        x_c, y_c, w_c, h_c = coords_c
                                        if x_c - x - w > 50:
                                            break
                                        else:
                                            compared_boxes.append(ele_c)
                                            text += " " + text_c
                                    segments.append(text)
                            print(f"{field_name}: {field_value}")
                            print(segments)
                            entry["Address Line 1"] = field_value
                            entry["Address Line 2"] = segments[0] if len(segments) == 6 else ""
                            entry["City"] = segments[-5]
                            entry["State"] = segments[-4]
                            entry["Zip"] = segments[-3]
                            entry["Number Type"] = segments[-2]
                            entry["Phone Number"] = segments[-1]
                        elif field_name in ["name", "admitting md"]:
                            f, m, l = name_parser(field_value)
                            print(f"{field_name}: {f}, {m} ,{l}")
                            if field_name == "name":
                                entry["First Name"] = f
                                entry["Middle Name"] = m
                                entry["Last Name"] = l
                            else:
                                entry["Admitting MD First Name"] = f
                                entry["Admitting MD Middle Name"] = m
                                entry["Admitting MD Last Name"] = l
                        else:
                            print(f"{field_name}: {field_value}")
                            entry[field_mapping[field_name]] = field_value
                df = df.append(entry, ignore_index=True)
            except:
                print("Phat gaya bhai")
                continue
            df.to_csv("test.csv", index=False)
            # break
        break


if __name__ == "__main__":
    main()
