import cv2

# image set-up(1920*1080)
image = cv2.imread('/home/jinhakim/anaconda3/envs/opencv_lec/Midterm_Project/bad_weather_images/tunnel/24_111849_220829_38.jpg')

# Reducing shadow
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
clahe_image[:, :, 0] = clahe.apply(clahe_image[:, :, 0])
shadow_reduced = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2BGR)

# Enhancing contrast
lab = cv2.cvtColor(shadow_reduced, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)
lab = cv2.merge((l, a, b))
contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Visualization of applying flitering
cv2.imshow('CLAHE Filter Applied Image', contrast_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
