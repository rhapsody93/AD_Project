import cv2

# image set-up(1920*1080)
image = cv2.imread('/home/jinhakim/anaconda3/envs/opencv_lec/Midterm_Project/bad_weather_images/tunnel/24_111849_220829_38.jpg')

# Converting image color(BGR) to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Converting equalized_image to BGR type
equalized_image_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# Visualization of applying filtering
cv2.imshow('Corrected Image', equalized_image_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
