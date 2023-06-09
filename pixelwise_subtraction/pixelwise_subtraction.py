# Pixel Wise Subtraction Functions

import numpy as np
import cv2
import matplotlib.cm as cm


def morph_erosion(img):
    """
    Do Morphological Erosion on the Image

    Parameters
    ----------
    img: Image to eroded

    Returns
    -------
    eroded_img: Image after morphological erosion
    """

    h, w = img.shape

    # Create a numpy array to store the eroded image
    eroded_img = np.zeros((h, w), np.uint8)

    for row in range(1, h - 1):#0 to h - 1
        for col in range(1, w - 1):
            # Get the 3x3 neighborhood of the current pixel. There will be three 3x3 matrices - one for each channel
            neighborhood = img[row-1:row+2, col-1:col+2]

            # Find the minimum value for each channel in the neighborhood. Three min values will be output.
            min_values = np.min(neighborhood, axis=(0, 1))

            # Set the values of the current pixel in the eroded image to the minimum values
            eroded_img[row, col] = min_values
    return eroded_img


def Pxl_score(original_image, normalized_image) -> float:
    """
    Calculate Pixel-wise subtraction score. Function also does morphological 
    erosion on image and normalizes the score from 1-7.

    Parameters
    ----------
    path1: Image path of first image

    path2: Image path of second image

    Returns
    -------
    score: Severity Rating Score
    """


    assert original_image.shape == normalized_image.shape, "Images must have the same dimensions"
    
    difference_image = cv2.absdiff(original_image, normalized_image)

    blue, green, red = cv2.split(difference_image)

    meroded_blue=morph_erosion(blue)
    meroded_green=morph_erosion(green)
    meroded_red=morph_erosion(red)

    merged_img=cv2.merge((meroded_blue, meroded_green, meroded_red))
    
    masksize=60866

    #Calculating the score
    squared_img = cv2.multiply(merged_img, merged_img)
    total_pixel_sum = cv2.sumElems(squared_img)
    total_pixel_sum=int(total_pixel_sum[0])
    mse=total_pixel_sum/masksize
    rmse= np.sqrt(mse)
    PxlSub = 1.15-(np.log10(rmse))
    score = 7*((PxlSub - 0) / (0.35 - 0))

    return score, merged_img

def heatmap_generation(eroded_image):
    image = eroded_image
    gray = np.mean(image, axis=2)
    gray_norm = gray / np.max(gray)
    heatmap = cm.jet(gray_norm)
    
    return heatmap

def pixelwise_subtraction_gradio(original_image, normalized_image):
    severity_rating, merged_img = Pxl_score(original_image=original_image, normalized_image=normalized_image)
    heatmap = heatmap_generation(merged_img)

    return (merged_img, heatmap, severity_rating)
