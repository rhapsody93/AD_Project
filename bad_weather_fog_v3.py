import cv2

# image set-up(1920*1080)
image = cv2.imread('/home/jinhakim/anaconda3/envs/opencv_lec/Midterm_Project/bad_weather_images/fog/16_072312_221021_08.jpg')

# Converting image color(BGR) to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Canny Edge Filtering
edges = cv2.Canny(gray_image, threshold1=30, threshold2=60)

# Visualization of applying filtering
cv2.imshow('Canny Edge Filtering', edges)
cv2.waitKey()
cv2.destroyAllWindows()

