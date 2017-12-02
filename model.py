import csv
import cv2
import numpy as np

# read the driving log file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# pop out the first row in the csv file
lines.pop(0)

# get the images and steering values
center_images = []
left_images = []
right_images = []
cam_images = []
steering_angles = []
for line in lines:
    # get the file path
    center_source_path = line[0]
    left_source_path = line[1]
    right_source_path = line[2]
    steering_angle = float(line[3])
    # get the images
    center_filename = center_source_path.split('/')[-1]
    left_filename = left_source_path.split('/')[-1]
    right_filename = right_source_path.split('/')[-1]
    center_current_path = 'data/IMG/'+ center_filename
    left_current_path = 'data/IMG/'+ left_filename
    right_current_path = 'data/IMG/'+ right_filename
    center_image = cv2.imread(center_current_path)
    left_image = cv2.imread(left_current_path)
    right_image = cv2.imread(right_current_path)
    center_images.append(center_image)
    left_images.append(left_image)
    right_images.append(right_image)
    cam_images.extend([center_image, left_image, right_image])
    # get the steering values
    correction = 0.2
    steering_angles.extend([steering_angle, steering_angle+correction, steering_angle-correction])
    
    
X_train = np.array(cam_images)
y_train = np.array(steering_angles)




# Data Processing

    