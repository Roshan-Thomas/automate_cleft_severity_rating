import gradio as gr

from imutils import face_utils
import imutils
import dlib 
import cv2


def mask_image(input_image):
    input_img = input_image
    mask = cv2.imread('images/mask.jpeg')
    mask = imutils.resize(mask, width=512)
    mask_inv = cv2.bitwise_not(mask)

    masked_img = cv2.bitwise_and(input_img, mask_inv)

    return masked_img


def detect_nasolabial_area(input_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("cnn_classifier/mouth_nose_predictor.dat")

    # read the input image
    input_image = input_image
    image = imutils.resize(input_image, width=512)

    masked_output_image = mask_image(image)

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (sX, sY) in shape:
            cv2.circle(image, (sX, sY), 1, (0, 0, 255), -1)
    
    classifier_output_image = image
    
    return (masked_output_image, classifier_output_image)


def classifier_ui():
    with gr.Column():
        with gr.Row():
            input_image = gr.Image()
            with gr.Column():
                masked_output_image = gr.Image(label="Image with Mask").style(height=300)
                classifier_ouput_image = gr.Image(label="Classifier Image").style(height=300)

        submit_button = gr.Button("Submit", variant="primary")
        submit_button.click(
            fn=detect_nasolabial_area,
            inputs=input_image,
            outputs=[masked_output_image, classifier_ouput_image],
        )
