import cv2
import numpy as np
import pytesseract
import pickle
import imutils
from align_images import align_images
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def show_image(image, window_name="untitled"):
    """
    Displays a given image in a named window (default name, 'Untitled')
    :param image: Image to be displayed
    :param window_name: Name of the window (default name, 'Untitled')
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def merge_into_rows(img, cnts):
    cnts, bounding_boxes = sort_contours(cnts)
    cnt_rows = []
    box_rows = []
    used_boxes = []
    cell_height = img.shape[0]/40
    # print(cell_height)
    for i, box in enumerate(bounding_boxes):
        if i in used_boxes:
            continue
        x, y, w, h = box
        if h > cell_height:
            continue
        box_row = [box]
        cnt_row = [cnts[i]]
        box_image = img[y:y + h, x:x + w]
        # print(i, box)
        # print(f"{y}:{y + h}, {x}:{x + w}")
        # print(pytesseract.image_to_string(box_image))
        # show_image(box_image)
        for j, box_c in enumerate(bounding_boxes[i+1:]):
            if abs(box[1] - box_c[1]) > cell_height//2.5:
                break
            # print(j, box_c)
            x, y, w, h = box_c
            box_c_image = img[y:y + h, x:x + w]
            # print(f"{y}:{y + h}, {x}:{x + w}")
            # print(pytesseract.image_to_string(box_c_image))
            # show_image(box_c_image)
            cnt_row.append(cnts[i+j+1])
            box_row.append(box_c)
            used_boxes.append(i+j+1)
        # print(used_boxes)
        (cnt_row, box_row) = zip(*sorted(zip(cnt_row, box_row),
                                             key=lambda b: b[1][0]))
        cnt_rows.append(cnt_row)
        box_rows.append(box_row)
        # print(box_rows)
    return cnt_rows, box_rows


def sort_contours(cnts):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                        key=lambda b: (b[1][1], b[1][0])))
    # return the list of sorted contours and bounding boxes
    return (cnts, bounding_boxes)


def process_image(img):
    # Denoising
    dst = img
    dst = cv2.fastNlMeansDenoising(dst, None, 20, 3, 21)
    return dst


def box_extraction(image):
    # # Thresholding the image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (thresh, img_bin) = cv2.threshold(image, 128, 255,
    #                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # Invert the image
    # img_bin = 255 - img_bin
    # cv2.imwrite("./tests/processing/Image_bin.jpg", img_bin)
    # # Defining a kernel length
    # kernel_length = np.array(image).shape[1] // 150
    # # A verticle kernel of (1 X kernel_length),
    # # which will detect all the verticle lines from the image.
    # verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
    #                                             (1, kernel_length))
    # # A horizontal kernel of (kernel_length X 1),
    # # which will help to detect all the horizontal line from the image.
    # hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # # A kernel of (3 X 3) ones.
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # Morphological operation to detect verticle lines from an image
    # img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    # verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # cv2.imwrite("./tests/processing/verticle_lines.jpg", verticle_lines_img)
    # # Morphological operation to detect horizontal lines from an image
    # img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    # horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    # cv2.imwrite("./tests/processing/horizontal_lines.jpg", horizontal_lines_img)
    # # Weighting parameters, this will decide the quantity
    # # of an image to be added to make a new image.
    # alpha = 0.5
    # beta = 1.0 - alpha
    # # This function helps to add two image with specific weight
    # # parameter to get a third image as summation of two image.
    # img_final_bin = cv2.addWeighted(verticle_lines_img, alpha,
    #                                 horizontal_lines_img,
    #                                 beta, 0.0)
    # img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    # (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255,
    #                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # For Debugging
    # # Enable this line to see verticle and horizontal lines
    # # in the image which is used to find boxes
    # cv2.imwrite("./tests/processing/img_final_bin.jpg", img_final_bin)
    # # Find contours for image, which will detect all the boxes
    # contours, hierarchy = cv2.findContours(
    #     img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # Sort all the contours by top to bottom.
    # # (contours, boundingBoxes) = sort_contours(contours)
    # (contours, boundingBoxes) = merge_into_rows(image, contours)
    # with open("bounding_boxes_1.pkl", "wb+") as file:
    #     pickle.dump(boundingBoxes, file)


    with open("bounding_boxes.pkl", "rb") as file:
        boundingBoxes = pickle.load(file)
    idx = 0
    images = []

    # finalBoundingBoxes = []
    # for row in boundingBoxes:
    #     row_box = []
    #     for b in row:
    #         x, y, w, h = b
    #         height, width = image.shape[:2]
    #         # if h > height / 5 or w > width / 2:
    #         #     continue
    #         idx += 1
    #         offset = 3
    #         y = max(y - offset, 0)
    #         x = max(x, 0)
    #         y_h = min(y + h + offset, height)
    #         x_w = min(x + w, width)
    #         new_img = image[y:y_h, x:x_w]
    #         skip_box = False
    #         for row in boundingBoxes:
    #             for c in row:
    #                 if c == b:
    #                     continue
    #                 x_c, y_c, w_c, h_c = c
    #                 if x_c > x and y_c > y and x_c + w_c < x + w and y_c + h_c < y + h:
    #                     skip_box = True
    #                     break
    #             if skip_box:
    #                 break
    #         if skip_box:
    #             continue
    #         cv2.rectangle(image, (x, y), (x_w, y_h), (0, 0, 255), 2)
    #         row_box.append(b)
    #     show_image(imutils.resize(image, width=500))
    #     finalBoundingBoxes.append(row_box)
    # with open("bounding_boxes_2.pkl", "wb+") as file:
    #     pickle.dump(finalBoundingBoxes, file)

    for i, row in enumerate(boundingBoxes):
        # print(f"Row {i}")
        row_image = []
        for b in row:
            # Returns the location and width,height for every contour
            x, y, w, h = b
            height, width = image.shape[:2]
            if h < height/90 or w < width/35:
                continue
            idx += 1
            offset = 3
            y = max(y-offset, 0)
            x = max(x, 0)
            y_h = min(y+h+offset, height)
            x_w = min(x+w, width)
            new_img = image[y:y_h, x:x_w]
            # show_image(new_img)
            cv2.rectangle(image, (x, y), (x_w, y_h), (0, 0, 255), 2)
            new_img = process_image(new_img)
            row_image.append(new_img)
        if row_image:
            images.append(row_image)
    show_image(imutils.resize(image, width=500))
    return images


# template = cv2.imread("./tests/template_2.jpg")
# image = cv2.imread("./tests/images/3.jpg")
# box_extraction(align_images(image, template))
