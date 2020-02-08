import cv2


def resize_image(input_image, new_image_shape):
    new_height, new_width = new_image_shape

    # OpenCV expects width and height in opposite order
    cv2_new_image_shape = (new_width, new_height)

    return cv2.resize(input_image, cv2_new_image_shape)
