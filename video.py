"""Yolo v3 video detection script.

Usage:
    python video.py <iou threshold> <confidence threshold> <filename>

Example:
    python video.py 0.5 0.5 YOLOv3/data/video/video01.mp4 # For video file input
    python video.py 0.5 0.5 0 # For cam input

Note: Only one video can be processed at one run.
"""

import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolov3 import YOLOv3Net
import cv2
import time
import sys

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416, 3)
number_of_classes = 80
class_name = 'YOLOv3/data/coco.names'
max_output_size = 100
max_output_size_per_class = 20

cfgfile = 'YOLOv3/configuration/yolov3.cfg'
weightfile = 'YOLOv3/weights/yolov3_weights.tf'

def main(iou_threshold, confidence_threshold, input_name):
    # Creates the model
    model = YOLOv3Net(cfgfile, model_size, number_of_classes)
    model.load_weights(weightfile)

    classes_names = load_class_names(class_name)

    window_name = 'Yolov3 video detection'
    cv2.namedWindow(winname=window_name)

    # Specify the video input.
    # 0 means input from cam 0.
    # For video, just change the 0 to video path
    video_input = 0
    if not str(input_name).isnumeric():
        video_input = input_name

    capture = cv2.VideoCapture(video_input)

    try:
        while True:
            start = time.time()
            
            # Reads the video stream
            ret, frame = capture.read()
            if not ret:
                break

            # Resizes the video frame to fit the window frame
            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            # Predicts the frame read
            prediction = model.predict(resized_frame)

            # Creates the output boxes where the model identified the classes whished
            boxes, scores, classes, numbers = output_boxes( \
                inputs=prediction, 
                model_size=model_size,
                max_output_size=max_output_size,
                max_output_size_per_class=max_output_size_per_class,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

            # Draws the boxes on the video frame
            image = draw_outputs(frame, boxes, scores, classes, numbers, classes_names)

            # Shows the video frame with the boxes
            cv2.imshow(window_name, image)

            stop = time.time()

            seconds = stop - start

            # Calculates the frames per second
            fps = 1 / seconds
            print("Estimated frames per second: {0}".format(fps))

            # If the 'q' key is pressed, the script stops 
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        capture.release()
        print('Detections have been performed successfully.')

if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])