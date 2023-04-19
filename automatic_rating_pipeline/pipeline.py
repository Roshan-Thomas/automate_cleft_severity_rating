from pixelwise_subtraction.pixelwise_subtraction import pixelwise_subtraction_gradio
from image_inpainting.ui import image_inpainting
from cnn_classifier.ui import detect_nasolabial_area
import cv2
import imutils
import numpy as np
from PIL import Image

def automatic_rating_pipeline(input_image):
    input_image = imutils.resize(input_image, width=512)
    input_shape = input_image.shape
    input_w, input_h = input_shape[0], input_shape[1]

    # classifier
    masked_nasolabial_area_image, _ = detect_nasolabial_area(input_image)

    print("[INFO] Identified Nasolabial Area.")

    # stable diffusion
    normalized_image = image_inpainting(input_image=input_image,
                                        mask_image=cv2.imread('images/mask.jpeg'),
                                        cfg_scale=7,
                                        seed=2633274231
                                        )

    print("[INFO] Completed Image Inpainting.")

    np_image = np.array(normalized_image)
    normalized_image_opencv = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    normalized_image = imutils.resize(normalized_image_opencv, width=input_w, height=input_h)

    # normalized_image = imutils.resize(normalized_image, width=512)

    # pixel wise subtraction
    _, _, severity_rating = pixelwise_subtraction_gradio(input_image, normalized_image)

    print("[INFO] Generated Severity Score")

    return ("{:.4f}".format(severity_rating-3), masked_nasolabial_area_image, normalized_image)