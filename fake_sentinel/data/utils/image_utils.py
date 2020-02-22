import os
import cv2


def read_image(filename):
    filename = str(filename)

    if not os.path.exists(filename):
        raise ValueError('{} does not exist'.format(filename))

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image(input_image, new_image_shape):
    new_height, new_width = new_image_shape

    # OpenCV expects width and height in opposite order
    cv2_new_image_shape = (new_width, new_height)

    return cv2.resize(input_image, cv2_new_image_shape)
