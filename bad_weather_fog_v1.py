import cv2
import matplotlib.pyplot as plt

# image set-up(1920*1080)
image = cv2.imread('/home/jinhakim/anaconda3/envs/opencv_lec/Midterm_Project/bad_weather_images/fog/16_072312_221021_08.jpg')

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Median Blur
median_blur = cv2.medianBlur(image, 5)

# Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Visualization of Applying filtering
#titles = ['original image', 'gaussian image', 'median image', 'bilateral image']
#images = [image, gaussian_blur, median_blur, bilateral_filter]

#for i in range(4):
#    plt.subplot(2, 2, i+1)
#    plt.imshow(images[i], cmap='gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])

#plt.tight_layout()
#plt.show()

cv2.imshow('Gaussian Filtered Image', gaussian_blur)
cv2.imshow('Median Filtered Image', median_blur)
cv2.imshow('Bilateral Filtered Image', bilateral_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()

