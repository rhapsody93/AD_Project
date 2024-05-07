import cv2
import numpy as np

# image set-up(1920*1080)
image = cv2.imread('/home/jinhakim/anaconda3/envs/opencv_lec/Midterm_Project/bad_weather_images/night/16_230314_220929_32.jpg')

# Applying Gaussian Blur to image
gaussian_blur = cv2.GaussianBlur(image, (7,7), 0)

# Applying Sharpening filter
sharpening = cv2.addWeighted(image, 2.8, gaussian_blur, -0.6, 0)

# Visualization of applying filtering
cv2.imshow('Filtering Applied Image', sharpening)
cv2.waitKey()
cv2.destroyAllWindows()

