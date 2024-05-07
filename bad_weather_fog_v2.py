import cv2
import numpy as np

# image set-up
image = cv2.imread('/home/jinhakim/anaconda3/envs/opencv_lec/Midterm_Project/bad_weather_images/fog/16_072312_221021_08.jpg')

# Converting image color(BGR) to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Laplacian filter to bad weather image(blurred)
laplacian_filtering = cv2.Laplacian(gray_image, cv2.CV_64F)

# Applying absolute value and converting type
laplacian_filtering = np.uint8(np.absolute(laplacian_filtering))

# Visualization of applying Laplacian filter
cv2.imshow('Laplacian Filter Applied Image', laplacian_filtering)
cv2.waitKey()
cv2.destroyAllWindows()
