import cv2

# Initializing the HOG detector for detecting pedestrians
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

# Loading the input video
cap = cv2.VideoCapture('/home/jinhakim/anaconda3/envs/opencv_lec/Final_Term_Project/pedestrians/pedestrians.mp4')

# Reading the frames of video
width = 800
ret, frame = cap.read()
height = int(frame.shape[0] * (width / frame.shape[1])) # height = image.shape[0], width = image.shape[1]
fps = cap.get(cv2.CAP_PROP_FPS)

# Setting the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('pedestrian_tracking.mp4', fourcc, fps, (width, height))

# Reading the frames of video while the input file is opened
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image_resized = cv2.resize(image, (width, height))

        # Detecting the regions including pedestrians
        (region, _) = hog.detectMultiScale(image_resized, winStride=(4, 4), padding=(4, 4), scale=1.05)

        # Drawing bounding boxes in the regions including pedestrians
        for (x, y, w, h) in region:
            cv2.rectangle(image_resized, (x, y),(x + w, y + h),(0, 255, 0), 2)

        # Writing the frame to the output video
        output.write(image_resized)

        # Visualization of result
        cv2.imshow('Original Video', cv2.resize(image, (width, height)))
        cv2.imshow('Pedestrians Tracking', image_resized)
        if cv2.waitKey(1) == 27:
            break

cap.release()
output.release()
cv2.destroyAllWindows()
