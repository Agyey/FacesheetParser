import cv2
import numpy as np
import os
import pandas as pd
import pytesseract
import easyocr
from pdf2image import convert_from_path
from ocr import process_image
from tqdm import tqdm
from pyjarowinkler import distance

import re

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


print("Loading Model...")
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
print("Model Loaded")


IMAGE_DIRECTORY = "facesheets"
CONFIG_DIRECTORY = "bin"
DEBUG = 0

junk = ['©']


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
    # cv2.destroyAllWindows()


def name_parser(text):
    """
    Extracts first, middle, last name from a text
    :param text: Text with name
    :return: first, middle, last name
    """
    first_name, middle_name, last_name = '', '', ''
    if "," in text or "." in text:
        text = text
        last_name, remaining_part = re.split("[,\.]", text)
        name_parts = [x.strip() for x in remaining_part.strip().split(" ") if x.strip()]
        if len(name_parts) == 2:
            first_name, middle_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
        else:
            first_name = last_name
            last_name = ''
    else:
        name_parts = text.strip().split(" ")
        if len(name_parts) == 3:
            first_name, middle_name, last_name = name_parts
        elif len(name_parts) == 2:
            first_name, last_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
    return first_name.strip().replace(";", ""), \
           middle_name.strip().replace(";", ""), \
           last_name.strip().replace(";", "")


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


def number_parser(text):
    number = text.strip()
    number = number.replace("o", "0")
    number = number.replace("O", "0")
    number = number.replace("l", "1")
    return number


def main():
    for file_name in tqdm(os.listdir('facesheets'), desc="Facesheets"):
        num_search = re.search("\d+(?=\.)", file_name)
        if num_search:
            file_num = int(num_search.group())
            if file_num < 13:
                continue
        else:
            continue
        df = pd.DataFrame(
            columns=[
                "First Name", "Middle Name", "Last Name",
                "Visit ID", "MRN", "DOB", "Age", "Gender", "SSN",
                "Admitting MD First Name", "Admitting MD Middle Name",
                "Admitting MD Last Name", "Address Line 1", "Address Line 2",
                "City", "State", "Zip", "Phone Number 1", "Phone Number 1 Type",
                "Phone Number 2", "Phone Number 2 Type",
                "Primary Policy Number", "Primary Plan Name",
                "Primary Insurance Address Line 1",
                "Primary Insurance Address Line 2", "Primary Insurance City",
                "Primary Insurance State", "Primary Insurance Zip",
                "Secondary Policy Number", "Secondary Plan Name",
                "Secondary Insurance Address Line 1",
                "Secondary Insurance Address Line 2", "Secondary Insurance City",
                "Secondary Insurance State", "Secondary Insurance Zip",
                "Tertiary Policy Number", "Tertiary Plan Name",
                "Tertiary Insurance Address Line 1",
                "Tertiary Insurance Address Line 2", "Tertiary Insurance City",
                "Tertiary Insurance State", "Tertiary Insurance Zip"
            ])
        sims = pd.DataFrame()
        images = convert_from_path(f'facesheets/{file_name}')
        c = 0
        for image in tqdm(images, desc="Pages", leave=False):
            c += 1
            entry = {}
            image.save("temp.jpg")
            result = reader.readtext('temp.jpg')
            fields = [
                "name", "visit #", "mr", "birthdate", "age", "gender", "ssn",
                "admitting md", "patient address/phone",
            ]
            field_mapping = {
                "visit #": "Visit ID", "mr": "MRN", "birthdate": "DOB",
                "age": "Age", "gender": "Gender",
                "ssn": "SSN",
            }
            assigned_fields = []
            for i, x in enumerate(result):
                # print(x)
                text = x[-2]
                sim_i = distance.get_jaro_distance(
                    text.lower(), "insurance", winkler=True, scaling=0.1)
                sim_p = distance.get_jaro_distance(
                    text.lower(), "primary", winkler=True, scaling=0.1)
                if (text.lower() == 'insurance' or sim_i >= 0.85) or\
                        (text.lower() == 'primary' or sim_p >= 0.85):
                    # print("Insurance Fields:")
                    insurance_fields = [
                        "insurance addr", "city", "state", "zip", "policy",
                        "cert #", "planname"
                    ]
                    field_mapping = {
                        "policy": "Policy Number", "planname": "Plan Name",
                        "cert #": "Policy Number",
                        "insurance addr": "Insurance Address Line 1",
                        "city": "Insurance City",
                        "state": "Insurance State", "zip": "Insurance Zip"
                    }
                    if not (text.lower() == 'primary' or sim_p >= 0.85):
                        assigned_insurance_fields = []
                        text = result[i + 1][-2]
                        sim_p = distance.get_jaro_distance(
                            text, "primary", winkler=True, scaling=0.1)
                    if sim_p > 0.85:
                        # print("Primary Insurance Found")
                        j = i + 1
                        text = result[j][-2]
                        sim_s = distance.get_jaro_distance(
                            text.lower(), "secondary", winkler=True, scaling=0.1)
                        sim_d = distance.get_jaro_distance(
                            text.lower(), "designated representative", winkler=True,
                            scaling=0.1)
                        while (text.lower() != "secondary"
                               and sim_s < 0.85
                               and text.lower() != "designated representative"
                               and sim_d < 0.85):
                            for field in insurance_fields:
                                if field in assigned_insurance_fields:
                                    continue
                                sim = distance.get_jaro_distance(
                                    text.lower(), field, winkler=True,
                                    scaling=0.1)
                                sims = sims.append(
                                    {'text': text.lower(), 'field': field,
                                     'sim': sim},
                                    ignore_index=True)
                                if text.lower().startswith(field) or \
                                        (sim >= 0.75 and abs(
                                            len(text) - len(field)) < 3):
                                    if field == "insurance addr":
                                        m = j + 1
                                        entry[
                                            "Primary Insurance Address Line 1"
                                        ] = result[m][-2]
                                        entry[
                                            "Primary Insurance Address Line 2"] = "NA"
                                        while m < len(result) and result[m][
                                            -2].lower().startswith("city"):
                                            m += 1
                                        if m - j == 3:
                                            entry[
                                                "Primary Insurance " \
                                                "Address Line 2"
                                            ] = result[m][-2]
                                        elif m - j == 2:
                                            entry[
                                                "Primary Insurance " \
                                                "Address Line 1"
                                            ] = "NA"
                                            entry[
                                                "Primary Insurance " \
                                                "Address Line 2"
                                            ] = "NA"
                                        # print(
                                        #     f"Primary Insurance "
                                        #     f"Address Line 1: "
                                        #     f"{entry['Primary Insurance Address Line 1']}")
                                        # print(
                                        #     f"Primary Insurance Address Line 2: "
                                        #     f"{entry['Primary Insurance Address Line 2']}")
                                    elif field == "planname":
                                        value = result[j + 1][-2]
                                        sim_score = distance.get_jaro_distance(
                                            value.lower(), "phone number", winkler=True,
                                            scaling=0.1)
                                        if result[j+1][-2] == "phone number" or sim_score >= 0.8:
                                            value = "NA"
                                        if ":" in result[j][-2]:
                                            first_part = result[j][-2].split(":")[-1]
                                            if first_part:
                                                value = first_part + (value if value != "NA" else "")
                                        # print(
                                        #     f"Primary {field}({text}): {value}")
                                        entry["Primary " + field_mapping[
                                            field]] = value
                                    elif field == "city":
                                        value = result[j+1][-2]
                                        sim_score = distance.get_jaro_distance(
                                            value.lower(), "state", winkler=True,
                                            scaling=0.1)
                                        if result[j+1][-2] == "state" or sim_score >= 0.8:
                                            value = "NA"
                                        # print(
                                        #     f"Primary {field}({text}): {value}")
                                        entry["Primary " + field_mapping[
                                            field]] = value
                                    elif field == "state":
                                        value = result[j+1][-2]
                                        sim_score = distance.get_jaro_distance(
                                            value.lower(), "zip", winkler=True,
                                            scaling=0.1)
                                        if result[j+1][-2] == "zip" or sim_score >= 0.8:
                                            value = "NA"
                                        # print(
                                        #     f"Primary {field}({text}): {value}")
                                        entry["Primary " + field_mapping[
                                            field]] = value
                                    elif field == "zip":
                                        value = result[j+1][-2]
                                        if not value.isnumeric():
                                            value = "NA"
                                        # print(
                                        #     f"Primary {field}({text}): {value}")
                                        entry["Primary " + field_mapping[
                                            field]] = value
                                    elif field == "cert #":
                                        if sim > 0.80:
                                            entry[
                                                "Primary Policy Number"
                                            ] = result[j - 1][-2]
                                            assigned_insurance_fields.append("policy")
                                            # print(
                                            #     f"Primary Policy "
                                            #     f"Number({text}): "
                                            #     f"{result[j-1][-2]}")
                                    else:
                                        value = result[j + 1][-2]
                                        # print(f"Primary {field}({text}): {value}")
                                        entry["Primary " + field_mapping[field]] = value
                            j += 1
                            if j >= len(result):
                                break
                            text = result[j][-2]
                            sim_s = distance.get_jaro_distance(
                                text, "secondary", winkler=True, scaling=0.1)
                            sim_d = distance.get_jaro_distance(
                                text, "designated representative", winkler=True,
                                scaling=0.1)
                        assigned_insurance_fields = []
                        if (text.lower() == "secondary" or sim_s > 0.85):
                            # print("Secondary Insurance Found")
                            k = j + 1
                            text = result[k][-2]
                            sim_t = distance.get_jaro_distance(
                                text.lower(), "tertiary", winkler=True, scaling=0.1)
                            sim_d = distance.get_jaro_distance(
                                text.lower(), "designated representative", winkler=True,
                                scaling=0.1)
                            while (text.lower() != "tertiary"
                                   and sim_t < 0.85
                                   and text.lower() != "designated representative"
                                   and sim_d < 0.85):
                                for field in insurance_fields:
                                    if field in assigned_insurance_fields:
                                        continue
                                    sim = distance.get_jaro_distance(
                                        text.lower(), field, winkler=True,
                                        scaling=0.1)
                                    sims = sims.append(
                                        {'text': text.lower(), 'field': field,
                                         'sim': sim},
                                        ignore_index=True)
                                    if text.lower().startswith(field) or \
                                            (sim >= 0.75 and abs(
                                                len(text) - len(field)) < 3):
                                        if field == "insurance addr":
                                            m = k + 1
                                            entry[
                                                "Secondary Insurance Address Line 1"] = \
                                                result[m][-2]
                                            entry[
                                                "Secondary Insurance Address Line 2"] = "NA"
                                            while m < len(result) and result[m][
                                                -2].lower().startswith("city"):
                                                m += 1
                                            if m - k == 3:
                                                entry[
                                                    "Secondary Insurance Address Line 2"] = \
                                                    result[m][-2]
                                            elif m - k == 2:
                                                entry[
                                                    "Secondary Insurance Address Line 1"] = "NA"
                                                entry[
                                                    "Secondary Insurance Address Line 2"] = "NA"
                                            # print(
                                            #     f"Secondary Insurance Address Line 1: "
                                            #     f"{entry['Secondary Insurance Address Line 1']}")
                                            # print(
                                            #     f"Secondary Insurance Address Line 2: "
                                            #     f"{entry['Secondary Insurance Address Line 2']}")
                                        elif field == "planname":
                                            value = result[k + 1][-2]
                                            sim_score = distance.get_jaro_distance(
                                                value.lower(), "phone number",
                                                winkler=True,
                                                scaling=0.1)
                                            if result[k + 1][
                                                -2] == "phone number" or sim_score >= 0.8:
                                                value = "NA"
                                            if ":" in result[k][-2]:
                                                first_part = \
                                                result[k][-2].split(":")[-1]
                                                if first_part:
                                                    value = first_part + (value if value != "NA" else "")
                                            # print(
                                            #     f"Secondary {field}({text}): {value}")
                                            entry["Secondary " + field_mapping[
                                                field]] = value
                                        elif field == "city":
                                            value = result[k + 1][-2]
                                            sim_score = distance.get_jaro_distance(
                                                value.lower(), "state",
                                                winkler=True,
                                                scaling=0.1)
                                            if result[k + 1][
                                                -2] == "state" or sim_score >= 0.8:
                                                value = "NA"
                                            # print(
                                            #     f"Secondary {field}({text}): {value}")
                                            entry["Secondary " + field_mapping[
                                                field]] = value
                                        elif field == "state":
                                            value = result[k + 1][-2]
                                            sim_score = distance.get_jaro_distance(
                                                value.lower(), "zip",
                                                winkler=True,
                                                scaling=0.1)
                                            if result[k + 1][
                                                -2] == "zip" or sim_score >= 0.8:
                                                value = "NA"
                                            # print(
                                            #     f"Secondary {field}({text}): {value}")
                                            entry["Secondary " + field_mapping[
                                                field]] = value
                                        elif field == "zip":
                                            value = result[k + 1][-2]
                                            if not value.isnumeric():
                                                value = "NA"
                                            # print(
                                            #     f"Secondary {field}({text}): {value}")
                                            entry["Secondary " + field_mapping[
                                                field]] = value
                                        elif field == "cert #":
                                            if sim > 0.80:
                                                entry["Secondary Policy Number"] = \
                                                    result[k - 1][-2]
                                                assigned_insurance_fields.append(
                                                    "policy")
                                                # print(
                                                #     f"Secondary Policy Number({text}): {result[k - 1][-2]}")
                                        else:
                                            value = result[k + 1][-2]
                                            # print(
                                            #     f"Secondary {field}({text}): {value}")
                                            entry["Secondary " + field_mapping[
                                                field]] = value
                                k += 1
                                if k >= len(result):
                                    break
                                text = result[k][-2]
                                sim_t = distance.get_jaro_distance(
                                    text, "tertiary", winkler=True,
                                    scaling=0.1)
                                sim_d = distance.get_jaro_distance(
                                    text, "designated representative",
                                    winkler=True,
                                    scaling=0.1)
                            assigned_insurance_fields = []
                            if (text.lower() == "tertiary" or sim_t > 0.85):
                                # print("Tertiary Insurance Found")
                                l = k + 1
                                text = result[l][-2]
                                sim_d = distance.get_jaro_distance(
                                    text.lower(), "designated representative",
                                    winkler=True,
                                    scaling=0.1)
                                while (text.lower() != "designated representative"
                                       and sim_d < 0.85):
                                    for field in insurance_fields:
                                        if field in assigned_insurance_fields:
                                            continue
                                        sim = distance.get_jaro_distance(
                                            text.lower(), field, winkler=True,
                                            scaling=0.1)
                                        sims = sims.append(
                                            {'text': text.lower(),
                                             'field': field,
                                             'sim': sim},
                                            ignore_index=True)
                                        if text.lower().startswith(field) or \
                                                (sim >= 0.75 and abs(
                                                    len(text) - len(
                                                        field)) < 3):
                                            if field == "insurance addr":
                                                m = l + 1
                                                entry[
                                                    "Tertiary Insurance Address Line 1"] = \
                                                    result[m][-2]
                                                entry[
                                                    "Tertiary Insurance Address Line 2"] = "NA"
                                                while m < len(result) and \
                                                        result[m][
                                                            -2].lower().startswith(
                                                            "city"):
                                                    m += 1
                                                if m - l == 3:
                                                    entry[
                                                        "Tertiary Insurance Address Line 2"] = \
                                                        result[m][-2]
                                                elif m - l == 2:
                                                    entry[
                                                        "Tertiary Insurance Address Line 1"] = "NA"
                                                    entry[
                                                        "Tertiary Insurance Address Line 2"] = "NA"
                                                # print(
                                                #     f"Tertiary Insurance Address Line 1: "
                                                #     f"{entry['Tertiary Insurance Address Line 1']}")
                                                # print(
                                                #     f"Tertiary Insurance Address Line 2: "
                                                #     f"{entry['Tertiary Insurance Address Line 2']}")
                                            elif field == "planname":
                                                value = result[l + 1][-2]
                                                sim_score = distance.get_jaro_distance(
                                                    value.lower(),
                                                    "phone number",
                                                    winkler=True,
                                                    scaling=0.1)
                                                if result[l + 1][
                                                    -2] == "phone number" or sim_score >= 0.8:
                                                    value = "NA"
                                                if ":" in result[l][-2]:
                                                    first_part = \
                                                    result[l][-2].split(":")[-1]
                                                    if first_part:
                                                        value = first_part + (value if value != "NA" else "")
                                                # print(
                                                #     f"Tertiary {field}({text}): {value}")
                                                entry[
                                                    "Tertiary " + field_mapping[
                                                        field]] = value
                                            elif field == "city":
                                                value = result[l + 1][-2]
                                                sim_score = distance.get_jaro_distance(
                                                    value.lower(), "state",
                                                    winkler=True,
                                                    scaling=0.1)
                                                if result[l + 1][
                                                    -2] == "state" or sim_score >= 0.8:
                                                    value = "NA"
                                                # print(
                                                #     f"Tertiary {field}({text}): {value}")
                                                entry["Tertiary " +
                                                      field_mapping[
                                                          field]] = value
                                            elif field == "state":
                                                value = result[l + 1][-2]
                                                sim_score = distance.get_jaro_distance(
                                                    value.lower(), "zip",
                                                    winkler=True,
                                                    scaling=0.1)
                                                if result[l + 1][
                                                    -2] == "zip" or sim_score >= 0.8:
                                                    value = "NA"
                                                # print(
                                                #     f"Tertiary {field}({text}): {value}")
                                                entry["Tertiary " +
                                                      field_mapping[
                                                          field]] = value
                                            elif field == "zip":
                                                value = result[l + 1][-2]
                                                if not value.isnumeric():
                                                    value = "NA"
                                                # print(
                                                #     f"Tertiary {field}({text}): {value}")
                                                entry["Tertiary " +
                                                      field_mapping[
                                                          field]] = value
                                            elif field == "cert #":
                                                if sim > 0.80:
                                                    entry[
                                                        "Tertiary Policy Number"] = \
                                                        result[l - 1][-2]
                                                    assigned_insurance_fields.append(
                                                        "policy")
                                                    # print(
                                                    #     f"Tertiary Policy Number({text}): {result[l - 1][-2]}")
                                            else:
                                                value = result[l + 1][-2]
                                                # print(
                                                #     f"Tertiary {field}({text}): {value}")
                                                entry["Tertiary " +
                                                      field_mapping[
                                                          field]] = value
                                    l += 1
                                    if l >= len(result):
                                        break
                                    text = result[l][-2]
                                    sim_d = distance.get_jaro_distance(
                                        text, "designated representative",
                                        winkler=True,
                                        scaling=0.1)
                    # print("Done Searching Insurance")
                    break
                for field in fields:
                    if field in assigned_fields:
                        continue
                    sim = distance.get_jaro_distance(
                        x[-2].lower(), field, winkler=True, scaling=0.1)
                    sims = sims.append(
                        {'text': x[-2].lower(), 'field': field, 'sim': sim},
                        ignore_index=True)
                    if x[-2].lower().startswith(field) or \
                            (sim >= 0.75 and abs(len(x[-2]) - len(field))<3):
                        # print(f"{field}({x[-2]})", end=": ")
                        if field == "patient address/phone":
                            j = i + 1
                            text = result[j][-2].lower()
                            sim_i = distance.get_jaro_distance(
                                text, "insurance", winkler=True, scaling=0.1)
                            sim_p = distance.get_jaro_distance(
                                text, "primary", winkler=True, scaling=0.1)
                            while text != 'insurance' and text != 'primary' and\
                                    sim_i < 0.85 and sim_p < 0.85:
                                # print(result[j], sim_i, sim_p)
                                text = result[j][-2].lower()
                                sim_i = distance.get_jaro_distance(
                                    text, "insurance", winkler=True,
                                    scaling=0.1)
                                sim_p = distance.get_jaro_distance(
                                    text, "primary", winkler=True,
                                    scaling=0.1)
                                # print(text, sim_i, sim_p)
                                j += 1
                            if text == 'primary' or sim_p >= 0.85:
                                j -= 1
                            segments = [x[-2] for x in result[i + 1:j-1]]
                            value = " ".join(segments)
                            # print(value)
                            second_line = True
                            if len(segments[2]) == 2:
                                second_line = False
                            entry["Address Line 1"] = segments[0]
                            offset = 0
                            if second_line:
                                entry["Address Line 2"] = segments[1]
                                offset = 1
                            else:
                                entry["Address Line 2"] = "NA"
                            entry["City"] = segments[offset + 1]
                            entry["State"] = segments[offset + 2]
                            entry["Zip"] = segments[offset + 3]
                            entry["Phone Number 1 Type"] = segments[offset + 4]
                            j = 5
                            while offset+j < len(segments) and segments[offset+j][0].isnumeric():
                                j += 1
                            entry["Phone Number 1"] = number_parser(
                                " ".join(segments[offset + 5:offset+j]))
                            if len(segments) - j > 1:
                                entry["Phone Number 2 Type"] = segments[offset + j]
                                entry["Phone Number 2"] = number_parser(
                                    " ".join(segments[offset + j + 1:]))
                        elif field == "insurance addr":
                            j = i + 1
                            entry["Insurance Address Line 1"] = result[j][-2]
                            while j < len(result) and result[j][-2].lower().startswith("city"):
                                j += 1
                            if j - i == 3:
                                entry["Insurance Address Line 2"] = result[j][-2]
                            elif j - i == 1:
                                entry["Insurance Address Line 1"] = "NA"
                                entry["Insurance Address Line 2"] = "NA"
                        elif field == "gender":
                            value = result[i + 1][-2]
                            if len(value) != 1:
                                value = "F"
                            else:
                                if value.lower() not in ['m', 'f']:
                                    value = "F"
                        #     x = result[i][0][0][0]
                        #     y = result[i][0][0][1]
                        #     x_w = result[i+1][0][0][0]
                        #     y_h = result[i+1][0][2][1]
                        #     new_img = np.array(image)[y-10:y_h+10, x+5:x_w-5]
                        #     text = pytesseract.image_to_string(new_img)
                        #     print(re.split("\s", text))
                        #
                        #     new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                        #     (thresh, img_bin) = cv2.threshold(new_img, 125, 255,
                        #                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        #     processed_image = process_image(new_img)
                        #     post_processed_image = process_image(img_bin)
                        #
                        #     print(pytesseract.image_to_string(new_img))
                        #     cv2.imwrite("temp.jpg", new_img)
                        #     print(re.split("\s", text))
                        #     show_image(new_img)
                        #
                        #     print(pytesseract.image_to_string(img_bin))
                        #     cv2.imwrite("temp.jpg", img_bin)
                        #     print(re.split("\s", text))
                        #     show_image(img_bin)
                        #
                        #     print(pytesseract.image_to_string(processed_image))
                        #     cv2.imwrite("temp.jpg", processed_image)
                        #     print(re.split("\s", text))
                        #     show_image(processed_image)
                        #
                        #     print(pytesseract.image_to_string(post_processed_image))
                        #     cv2.imwrite("temp.jpg", post_processed_image)
                        #     print(re.split("\s", text))
                        #     show_image(post_processed_image)
                        else:
                            if i + 1 < len(result):
                                value = result[i+1][-2]
                                if not value.strip():
                                    value = "NA"
                                if field in ["name", "admitting md"]:
                                    f, m, l = name_parser(value)
                                    value = f"{f}, {m} ,{l}"
                                    if field == "name":
                                        entry["First Name"] = f
                                        entry["Middle Name"] = m
                                        entry["Last Name"] = l
                                    else:
                                        entry["Admitting MD First Name"] = f
                                        entry["Admitting MD Middle Name"] = m
                                        entry["Admitting MD Last Name"] = l
                                elif ":" in value or "#" in value:
                                    value = x[-2].split(":")[-1]
                                if field not in ["name", "admitting md"] and field in field_mapping:
                                    entry[field_mapping[field]] = value
                                # print(value)
                        assigned_fields.append(field)
                        break
            df = df.append(entry, ignore_index=True)
            df.to_csv(f"test_{file_num}.csv", index=False)
            sims.to_csv(f"sims_{file_num}.csv", index=False)


if __name__ == "__main__":
    main()
