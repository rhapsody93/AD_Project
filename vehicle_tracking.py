import cv2

# Initializing the Haar Cascade detector(pre-trained model) for detecting vehicles
haar_vehicle_cascade = cv2.CascadeClassifier('cars.xml')

# cars.xml : https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml

# Loading the input video
cap = cv2.VideoCapture('/home/jinhakim/anaconda3/envs/opencv_lec/Final_Term_Project/dynamic_object/videos_train_00005.mp4')
width = 800
ret, frame = cap.read()
height = int(frame.shape[0] * (width / frame.shape[1])) # height = image.shape[0], width = image.shape[1]
fps = cap.get(cv2.CAP_PROP_FPS)

# Setting the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('vehicle_tracking.mp4', fourcc, fps, (width, height))

# Reading the frames of video while the input file is opened
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image_resized = cv2.resize(image, (width, height))

        # Detecting the regions including vehicles
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        vehicle_regions = haar_vehicle_cascade.detectMultiScale(gray_image, 1.3, 3)

        # Drawing bounding boxes in the regions including vehicles
        for (x, y, w, h) in vehicle_regions:
            cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Writing the frame to the output video
        output.write(image_resized)

        # Visualization of result
        cv2.imshow('Original Video', cv2.resize(image, (width, height)))
        cv2.imshow('Dynamic Objects Tracking', image_resized)
        if cv2.waitKey(100) == 27:
            break

cap.release()
output.release()
cv2.destroyAllWindows()