from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# read input image file
image_path = args["image"]
image = cv2.imread(image_path)

# resize the image for faster processing
image = imutils.resize(image, width=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 0)

# loop over the face detections
for rect in rects:
    # convert the dlib rectangle into an OpenCV bounding box and
    # draw a bounding box surrounding the face
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

    # use our custom dlib shape predictor to predict the location
    # of our landmark coordinates, then convert the prediction to
    # an easily parsable Numpy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the (x,y)-coordinates from our dlib shape
    # predictor model draw them on the image
    for (sX, sY) in shape:
        cv2.circle(image, (sX, sY), 1, (0,0,255), -1)

# write output image file
output_path = os.path.splitext(image_path)[0] + '_output.jpg'
cv2.imwrite(output_path, image)

# display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()