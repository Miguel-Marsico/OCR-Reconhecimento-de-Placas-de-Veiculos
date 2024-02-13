# Import necessary libraries
import cv2
import numpy as np
import imutils
import easyocr

# Load the image
img = cv2.imread('unnamed.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter for edge smoothing
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Apply adaptive thresholding to enhance the edges
thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Detect edges using the Canny edge detector
edged = cv2.Canny(thresh, 30, 200)

# Find contours in the edged image
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

# Sort contours based on contour area, keeping the top 10
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Initialize a variable to hold the location of the license plate
location = None

# Loop through the contours to find a contour with 4 corners (rectangle)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# If a rectangular contour was found
if location is not None:
    # Create a mask for the detected region
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Crop the region from the image
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    # Optionally, apply histogram equalization to improve the cropped image contrast before OCR
    cropped_image = cv2.equalizeHist(cropped_image)

    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])

    # Perform OCR on the cropped image
    result = reader.readtext(cropped_image)

    # If OCR results were found
    if result:
        text = result[0][-2]

        # Display the detected text on the original image
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                          color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

        # Save the result
        cv2.imwrite('output.jpg', img)
    else:
        print("Error")
