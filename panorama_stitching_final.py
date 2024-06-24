import cv2
import numpy as np

# Cylindrical Projection
def cylindrical_projection(image, f):
    h, w = image.shape[:2]
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # Camera intrinsic parameter matrix

    # Creating cylindrical coordinates
    cylinder = np.zeros_like(image)
    cyl_x, cyl_y = np.meshgrid(np.arange(w), np.arange(h))
    cyl_x = cyl_x - w / 2
    cyl_y = cyl_y - h / 2
    theta = np.arctan(cyl_x / f)
    h_ = np.sqrt(cyl_x ** 2 + f ** 2)
    y_ = cyl_y * f / h_
    x_ = f * np.tan(cyl_x / f)

    xmap = x_ + w / 2
    ymap = y_ + h / 2

    # Performing cylindrical projection using remapping
    cylinder = cv2.remap(image, xmap.astype(np.float32), ymap.astype(np.float32), cv2.INTER_LINEAR)

    return cylinder

# Cropping the contours(black borders) of panorama video
def crop_black_borders(image):
    # Converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Finding contours of the non-black regions
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Finding the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Cropping the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

video_files = ['/home/jinhakim/anaconda3/envs/opencv_lec/Final_Term_Project/left.mp4',
               '/home/jinhakim/anaconda3/envs/opencv_lec/Final_Term_Project/right.mp4']

frames_list = []
frame_interval = 5
resize_num = 0.4

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f'Error has occurred during opening : {video_file}')
        continue

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    print(f'Number of frames extracted from {video_file} : {len(frames)}')

    frames_list.append(frames)

min_frame_count = min(len(frames) for frames in frames_list)
frames_list = [frames[:min_frame_count] for frames in frames_list]

output_file = '/home/jinhakim/anaconda3/envs/opencv_lec/Final_Term_Project/panorama_stitching.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10

first_frame = frames_list[0][0]
height, width, _ = first_frame.shape
frame_size = (int(width*resize_num), int(height*resize_num))
panorama_size = (frame_size[0]*2, frame_size[1])

output = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Initializing estimate for focal length
f0 = width*0.8

for i in range(min_frame_count):
    frames_to_stitch = [frames_list[j][i] for j in range(len(frames_list))]

    stitcher = cv2.createStitcher()
    ret, panorama = stitcher.stitch(frames_to_stitch)

    if ret == cv2.STITCHER_OK:
        panorama_resized = cv2.resize(panorama, frame_size)
        output.write(panorama_resized)

        left_frame = frames_list[0][i]
        right_frame = frames_list[1][i]

        left_text_pos = (left_frame.shape[1] // 2 - 20, left_frame.shape[0] - 30)
        right_text_pos = (right_frame.shape[1] // 2 - 20, right_frame.shape[0] - 30)

        cv2.putText(left_frame, 'Left', left_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(right_frame, 'Right', right_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

        combined_frame = np.hstack((left_frame, right_frame))
        combined_frame_resized = cv2.resize(combined_frame, (frame_size[0], frame_size[1]))

        # Visualization of original videos(left and right)
        cv2.imshow('Original Videos', combined_frame_resized)

        panorama_cylindrical = cylindrical_projection(panorama_resized, f=f0)

        # Visualization of panorama video
        cropped_panorama = crop_black_borders(panorama_cylindrical)
        cv2.imshow('Panorama Video', cropped_panorama)

        if cv2.waitKey(100) == 27:
            break

    else:
        print(f'Error has occurred during stitching frame {i}')
        continue

output.release()
cv2.destroyAllWindows()
print('Panorama stitching has been created successfully.')
